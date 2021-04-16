from PIL import Image
from matplotlib import pyplot as plt, dates, ticker
import numpy as np
from os.path import sep

city_names = {
    'sf': 'San Francisco',
    'mw': 'Milwaukee',
    'br': 'Baton Rouge',
    'nola': 'New Orleans'
}


def plot_c4(city, crime, crime_timeline, covid, covid_timeline, c4, c4_domain, crime_type='all', tract='all',
            auto_save=False, show_on_save=False, legend=True):

    if city != 'br':
        fig, (ax_crime, ax_covid, ax_corr) = plt.subplots(3, 1, figsize=(7, 4.8))

        title = '{} Crime-Covid Cross Correlation'.format(city_names[city])
        plt.suptitle(title)

        subtitle = ''
        if crime_type != 'all':
            subtitle += 'crime type: {}'.format(crime_type)

        if tract != 'all':
            subtitle += '    tract: {}'.format(str(tract))

        plt.title(subtitle, fontsize=7)

        ax_crime.set_ylabel('# of incidents')

        m = 12
        xticks = ticker.MaxNLocator(m)
        ax_crime.xaxis.set_major_locator(xticks)

        ax_crime.plot(crime_timeline, crime, c='green', label='all crime' if crime_type == 'all' else crime_type)

        ax_covid.plot(covid_timeline, covid, c='red', label='new reports of covid' if tract == 'all' else 'Covid in ' + str(tract))
        ax_covid.xaxis.set_major_locator(xticks)

        ax_corr.plot(c4_domain, c4, c='magenta', label='correlation')

        if legend:
            ax_crime.legend()
            ax_covid.legend()
            ax_corr.legend()
    else:
        fig, (ax_orig, ax_corr) = plt.subplots(2, 1, figsize=(7, 4.8))

        title = '{} Crime-Covid Cross Correlation'.format(city_names[city])
        plt.suptitle(title)

        subtitle = ''
        if crime_type != 'all':
            subtitle += 'crime type: {}'.format(crime_type)

        if tract != 'all':
            subtitle += '    tract: {}'.format(str(tract))

        plt.title(subtitle, fontsize=7)

        ax_orig.set_ylabel('# of incidents')

        m = 12
        xticks = ticker.MaxNLocator(m)
        ax_orig.xaxis.set_major_locator(xticks)

        ax_orig.plot(crime_timeline, crime, c='green', label='all crime' if crime_type == 'all' else crime_type)

        ax_orig.plot(covid_timeline, covid, c='red',
                      label='new reports of covid' if tract == 'all' else 'Covid in ' + str(tract))

        ax_corr.plot(c4_domain, c4, c='magenta', label='correlation')

        if legend:
            ax_orig.legend()
            ax_corr.legend()

    plt.show()


def plot_crime_arima(city, crime_type, train, val, test, predictions, past_crime, covid_crime, all_crime_monthly,
                     val_performance=None, test_performance=None, model_params=None,
                     one_month=30, conf_int=None, plot_real_months=False, test_covid=False,
                     auto_save=False, show_on_save=False, legend=False):
    val_size = val.shape[0]
    train_size = train.shape[0]

    # Generate domains for plots ##
    months_x = np.arange(all_crime_monthly.shape[0], step=1)
    months_x *= one_month

    days_x = np.arange(past_crime.shape[0], step=1)
    all_days_x = np.arange(past_crime.shape[0] + covid_crime.shape[0], step=1)

    # Plot ##
    plt.figure(figsize=(7, 4))

    subtitle = ''
    if test_covid and test_performance is not None:
        subtitle += 'test performance integral = {}, maximum distance between test and forecast = {}\n'.format(
            test_performance[0], test_performance[1])

    if val_performance is not None:
        subtitle += 'validation performance integral = {}, maximum distance between validation and forecast = {}'. \
            format(val_performance[0], val_performance[1])

    plt.title(subtitle, fontsize=7)

    plt.xlabel('Days since Janurary 1, 2018')
    plt.ylabel('# of criminal occurances')

    end = months_x.shape[0]

    # Training / Validation in days
    plt.plot(days_x, past_crime, c='orange', label='daily model training/validation crime data')

    # Smoothed Training / Validation in months
    plt.plot(months_x[:train_size], train, c='blue', label='smoothed monthly training data')
    plt.plot(months_x[train_size:train_size + val_size], val, c='yellow', label='smoothed monthly validation data')

    # Maximum validation difference
    max_val_ind = val_performance[2]
    if max_val_ind != -1:
        max_diff_y = [val[max_val_ind], predictions[:val_size][max_val_ind]]
        max_diff_x = [months_x[train_size:train_size + val_size][max_val_ind],
                      months_x[train_size:train_size + val_size][max_val_ind]]

        plt.plot(max_diff_x, max_diff_y, c='black', label='maximum difference')

    # Maximum test difference
    if test_performance is not None and test_performance[2] != -1:
        max_val_ind = test_performance[2]
        max_diff_y = [test[max_val_ind], predictions[val_size:][max_val_ind]]
        max_diff_x = [months_x[train_size + val_size:][max_val_ind],
                      months_x[train_size + val_size:][max_val_ind]]

        plt.plot(max_diff_x, max_diff_y, c='black')

    # Covid data daily / monthly
    if test_covid:
        plt.plot(all_days_x[-covid_crime.shape[0]:], covid_crime, c='magenta', label='daily covid crime data')
        plt.plot(months_x[train_size + val_size:], all_crime_monthly[train_size + val_size:], c='purple',
                 label='smoothed monthly covid data (saglov with train)')
        plt.plot(months_x[train_size + val_size:],
                 test[:-1] if len(test) > len(months_x[train_size + val_size:]) else test[:], c='red',
                 label='smoothed monthly covid data')

    # Model predictions & confidence interval
    if not test_covid:
        end -= test.shape[0]

    plt.plot(months_x[train_size:end], predictions[:end - train_size], c='green', label='monthly model forecast')
    if conf_int is not None:
        plt.plot(months_x[train_size:end], conf_int.T[0][:end - train_size], c='gray', label='95% confidence interval')
        plt.plot(months_x[train_size:end], conf_int.T[1][:end - train_size], c='gray')

    # Plot real monthly data
    if plot_real_months:
        # real_months, real_monthly_crime = month_sum(all_crime, timeline)
        # plt.plot(real_months, real_monthly_crime, c='brown', label='real monthly crime')
        # plt.gcf().autofmt_xdate()
        pass

    # Format title
    crime_type_words = crime_type.split(' ')
    caps_crime_type = ''
    for i, c in enumerate(crime_type_words):
        caps_crime_type += c[0].upper() + c[1:].lower()
        caps_crime_type += '' if i == len(crime_type_words) - 1 else ' '

    title = '{} {} Crime Data & Model Forecast'.format(city_names[city], caps_crime_type)
    if model_params is not None:
        title += '\nmodel order = {}, seasonal order = {}'.format(model_params['order'],
                                                                  model_params['seasonal_order'])

    plt.suptitle(title)
    if legend:
        plt.legend()

    # Save
    if auto_save:
        impath = sep.join(['..', 'plots', city, crime_type + ('test' if test_covid else 'val') + '.png'])
        plt.savefig(impath, orientation='landscape', format='png', dpi=300)
        if show_on_save:
            Image.open(impath).show()
    else:
        plt.show()
