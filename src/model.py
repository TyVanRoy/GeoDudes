import numpy as np
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from scipy.signal import savgol_filter

gen_path = 'gen'


def evaluate_predictions(predictions, reality, conf_int=None):

    max_diff = 0
    max_dif_ind = -1
    integral = 0

    high_conf_int = 0
    low_conf_int = 0

    for i, v in enumerate(reality):
        diff = v - predictions[i]
        integral += diff

        if abs(diff) > max_diff:
            max_diff = abs(diff)
            max_dif_ind = i

        if conf_int is not None:
            high_diff = v - conf_int[i][1]
            low_diff = conf_int[i][0] - v

            if high_diff > 0:
                high_conf_int += high_diff
            if low_diff > 0:
                low_conf_int += low_diff

    if conf_int is None:
        return integral, max_diff, max_dif_ind
    else:
        return integral, max_diff, max_dif_ind, high_conf_int, low_conf_int


def month_from(date):
    return '/'.join(date.split('/')[i] for i in [0, 2])


def month_sum(data, timeline):
    current_month = month_from(timeline[0])
    months = [current_month]
    monthly_data = [0]
    for i, d in enumerate(data):
        month = month_from(timeline[i])
        if month != current_month:
            current_month = month
            months.append(current_month)
            monthly_data.append(0)

        monthly_data[len(months) - 1] += d

    return months, monthly_data


def even_samples(curve, interval):
    new_curve = []
    for i, p in enumerate(curve):
        if i % interval == 0:
            new_curve.append(p)
    return np.asarray(new_curve)


def generate_arima_model(data, covid_days, val_months, m=1, month=30, year=365, poly_order=3,
                         smooth_together=False, return_metacrime=False):

    all_crime = data

    # Smooth and sample monthly, then split into past_crime_monthly and covid_crime_monthly ##
    one_month = month
    one_year = year

    all_crime_smooth = savgol_filter(all_crime, one_year, poly_order) 
    all_crime_monthly = even_samples(all_crime_smooth, one_month)

    month_split_ind = int(all_crime.shape[0] - covid_days / one_month)

    past_crime, covid_crime = train_test_split(all_crime, train_size=int(all_crime.shape[0] - covid_days))

    if smooth_together:
        print('\t\t*** Smoothing together...')
        past_crime_monthly = all_crime_monthly[:month_split_ind]
        covid_crime_monthly = all_crime_monthly[month_split_ind:]
    else: # The right way
        past_crime_smooth = savgol_filter(past_crime, one_year, poly_order) 
        past_crime_monthly = even_samples(past_crime_smooth, one_month)

        covid_crime_smooth = savgol_filter(covid_crime, one_year, poly_order)  
        covid_crime_monthly = even_samples(covid_crime_smooth, one_month)

    # Split training and validation from past ##
    months = past_crime_monthly.shape[0]
    val_size = val_months
    train_size = months - val_size
    train, val = train_test_split(past_crime_monthly, train_size=train_size)
    test = covid_crime_monthly

    # Estimate model parameters ##
    model = pm.auto_arima(train, seasonal=True, m=m)

    if return_metacrime:
        return model, train, val, test, past_crime, covid_crime, all_crime_monthly
    else:
        return model, train, val, test