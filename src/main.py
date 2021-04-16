from GeoSeries import *
from model import *
from plot import *
from tqdm import tqdm
from os.path import sep
from scipy.signal import correlate, correlation_lags

data_path = 'data'
gen_path = 'gen'
crime_ext = '_crime.csv'
covid_ext = '_covid.csv'

city_splits = {
    'sf': 395,
    'mw': 387,
    'br': 323,
    'nola': 381
}

shared_types = ['burglary', 'robbery', 'theft']

city_params = dict(sf={'assault': 1,
                       'burglary - residential': 2,
                       'burglary - commercial': 1,
                       'larceny theft': 3,
                       'motor vehicle theft': 2,
                       'robbery': 1,
                       'domestic violence': 1,
                       'burglary': 2,
                       'theft': 5,
                       'all': 2},
                   mw={'theft': 1,
                       'sex_crime': 1,
                       'assault': 1,
                       'burglary': 1,
                       'robbery': 1,
                       'all': 1},
                   br={'assault': 1,
                       'theft': 4,
                       'vehicle burglary': 3,
                       'residential burglary': 1,
                       'non-residential burglary': 1,
                       'individual robbery': 1,
                       'domestic violence': 6,
                       'business robbery': 1,
                       'burglary': 3,
                       'robbery': 1,
                       'all': 1},
                   nola={'theft': 1,
                         'assault': 1,
                         'domestic violence': 1,
                         'burglary': 1,
                         'robbery': 1,
                         'all': 1})


def plot_arima(city, covid_days, val_months, test_covid=False, month='30', year='365', poly_order=3, auto_save=False):
    # Load crime data ##
    crime_by_type, crime_by_tract, types, tracts, crime_timeline = load_data(city)
    print(crime_by_type.shape)

    ms = city_params[city]

    for i, crime_type in enumerate(types):
        crime_type = crime_type.lower()
        model, train, val, test, past_crime, covid_crime, all_crime_monthly = \
            generate_arima_model(crime_by_type[i], covid_days, val_months, m=ms[crime_type],
                                 month=month, year=year, poly_order=poly_order,
                                 return_metacrime=True)

        # Get model parameters ##
        all_model_params = model.get_params()
        model_params = {'order': all_model_params['order'],
                        'seasonal_order': all_model_params['seasonal_order']}

        # Get predictions ##
        predictions, conf_int = model.predict(val.shape[0] + test.shape[0], return_conf_int=True)

        # Evaluate validation predictions ##
        val_performance = evaluate_predictions(predictions[:val.shape[0]], val)

        # Evaluate test predictions ##
        test_performance = None
        if test_covid:
            test_performance = evaluate_predictions(predictions[val.shape[0]:], test)

        plot_crime_arima(city, crime_type, train, val, test, predictions, past_crime, covid_crime,
                         all_crime_monthly, val_performance=val_performance, test_performance=test_performance,
                         conf_int=conf_int, model_params=model_params,
                         one_month=30, plot_real_months=False, test_covid=test_covid, auto_save=auto_save,
                         legend=True if crime_type == 'all' else False)


def find_best_m(data, covid_days, val_months, month, year, poly_order, max_m=9):
    min_integral = float('inf')
    best_m = -1

    for chosen_m in tqdm(range(1, max_m + 1)):
        try:
            model, train, val, test = generate_arima_model(data, covid_days, val_months,
                                                           m=chosen_m,
                                                           month=month, year=year,
                                                           poly_order=poly_order,
                                                           return_metacrime=False)
        except ValueError:
            print('Value error for m=' + str(chosen_m))
            continue

        # Get predictions ##
        predictions, conf_int = model.predict(val.shape[0] + test.shape[0], return_conf_int=True)

        # Evaluate validation predictions ##
        integral, max_diff, max_diff_ind = evaluate_predictions(predictions[:val.shape[0]], val)

        if abs(integral) < min_integral:
            min_integral = integral
            best_m = chosen_m

    return best_m


def hole_check(shapes):
    print('HOLE CHECK: ', end='')
    shape = shapes[0]
    for s in shapes[1:]:
        if s != shape:
            print('FAILED\nexiting program...')

    print('PASSED')


def gen_data(city):
    # Crime #

    # Load data from csv ##
    crime_data = CrimeSeries(sep.join(['..', data_path, city + crime_ext]))
    type_map = crime_data.get_fmap('type')
    tract_map = crime_data.get_fmap('tract_id')

    # Print data ##
    GeoSeries.print_fmap(type_map)
    GeoSeries.print_fmap(tract_map)

    # Build crime-by-type series ##
    types = type_map.T[0].tolist()

    crime_by_type = []
    shapes = []
    print('Building crime-by-type series...')
    for i, t in enumerate(tqdm(types)):
        type_series = crime_data.get_fmap('date', filterBy='type', filters=[t], loadbar=False)
        if i == 0:
            crime_timeline = type_series.T[0]
        type_series = type_series.T[1]
        type_series = list(map(float, type_series))
        crime_by_type.append(type_series)
        shapes.append(np.asarray(type_series).shape)

    hole_check(shapes)

    # Add a summed data col ##
    all_series = np.asarray(crime_by_type)
    all_series = np.sum(all_series, axis=0)

    # Add merged similar type cols ##
    for t in shared_types:
        if len(get_type_indices(t, types)) > 1:
            print("Adding merged {} data...".format(t))
            crime_by_type, types = add_merged_types(t, types, crime_by_type)

    for i, t in enumerate(types):
        types[i] = types[i].lower()

    crime_by_type.append(all_series)
    types.append('all')

    # Build crime-by-tract series ##
    tracts = tract_map.T[0].tolist()

    crime_by_tract = []
    shapes = []
    print('Building crime-by-tract series...')
    for i, t in enumerate(tqdm(tracts)):
        tract_series = crime_data.get_fmap('date', filterBy='tract_id', filters=[t], loadbar=False)
        tract_series = tract_series.T[1]
        tract_series = list(map(float, tract_series))
        crime_by_tract.append(tract_series)
        shapes.append(np.asarray(tract_series).shape)

    hole_check(shapes)

    # Covid #

    covid_data = CovidSeries(sep.join(['..', data_path, city + covid_ext]))

    # Build covid-by-tract series ##
    tracts = tracts

    covid_by_tract = []
    shapes = []
    print('Building covid-by-tract series...')
    for i, t in enumerate(tqdm(tracts)):
        tract_series = covid_data.get_fmap([t], loadbar=False, alt_index=(city == 'nola' or city == 'br'))
        if i == 0:
            covid_timeline = tract_series.T[0]
        tract_series = tract_series.T[1]
        tract_series = list(map(float, tract_series))
        covid_by_tract.append(tract_series)
        shapes.append(np.asarray(tract_series).shape)

    hole_check(shapes)

    crime_by_type = np.asarray(crime_by_type)
    crime_by_tract = np.asarray(crime_by_tract)
    types = np.asarray(types)
    tracts = np.asarray(tracts)

    covid_by_tract = np.asarray(covid_by_tract)

    print('crime_by_tract, crime_by_type, crime_timeline')
    print(crime_by_tract.shape)
    print(crime_by_type.shape)
    print(crime_timeline.shape)
    print(types)
    print(tracts)

    print('covid_by_tract, covid_timeline')
    print(covid_by_tract.shape)
    print(covid_timeline.shape)

    # Special case for mw
    if city == 'mw':
        crime_by_type = crime_by_type[:, :-5]
        print("Cutting off last 5 days of Milwaukee")
        print(crime_by_type.shape)

    np.save(sep.join(['..', gen_path, city + '_crime_by_type.npy']), crime_by_type)
    np.save(sep.join(['..', gen_path, city + '_crime_by_tract.npy']), crime_by_tract)
    np.save(sep.join(['..', gen_path, city + '_types.npy']), types)
    np.save(sep.join(['..', gen_path, city + '_tracts.npy']), tracts)
    np.save(sep.join(['..', gen_path, city + '_crime_timeline.npy']), crime_timeline)

    np.save(sep.join(['..', gen_path, city + '_covid_by_tract.npy']), covid_by_tract)
    np.save(sep.join(['..', gen_path, city + '_covid_timeline.npy']), covid_timeline)


def load_data(city, load_covid=False):
    crime_by_type = np.load(sep.join(['..', gen_path, city + '_crime_by_type.npy']))
    crime_by_tract = np.load(sep.join(['..', gen_path, city + '_crime_by_tract.npy']))
    types = np.load(sep.join(['..', gen_path, city + '_types.npy']))
    tracts = np.load(sep.join(['..', gen_path, city + '_tracts.npy']))
    crime_timeline = np.load(sep.join(['..', gen_path, city + '_crime_timeline.npy']))

    if not load_covid:
        return crime_by_type, crime_by_tract, types, tracts, crime_timeline

    covid_by_tract = np.load(sep.join(['..', gen_path, city + '_covid_by_tract.npy']))
    covid_timeline = np.load(sep.join(['..', gen_path, city + '_covid_timeline.npy']))

    return crime_by_type, crime_by_tract, types, tracts, crime_timeline, covid_by_tract, covid_timeline


def generate_arima_params(city, val_months, month, year, poly_order, max_m=9):
    covid_days = city_splits[city]
    crime_by_type, crime_by_tract, types, tracts, crime_timeline = load_data(city)

    params = {}

    for i, crime_type in enumerate(types):
        crime_type = crime_type.lower()
        print('Finding best m for {}...'.format(crime_type))
        params[crime_type] = find_best_m(crime_by_type[i], covid_days, val_months, month, year, poly_order, max_m=max_m)

    return params


def get_type_indices(crime_type, types):
    matches = []
    for i, t in enumerate(types):
        if crime_type in t.lower():
            matches.append(i)
    return matches


def add_merged_types(crime_type, types, by_type_crime_data):
    indexes = get_type_indices(crime_type, types)
    to_combine = []
    for i in indexes:
        to_combine.append(by_type_crime_data[i])

    to_combine = np.asarray(to_combine)
    to_combine = np.sum(to_combine, axis=0)

    by_type_crime_data.append(to_combine)
    types.append(crime_type)

    return by_type_crime_data, types


def multi_plot_c4(city):
    crime_by_type, crime_by_tract, types, tracts, crime_timeline, covid_by_tract, covid_timeline = load_data(city, load_covid=True)

    all_crime = crime_by_type[len(types) - 1]
    all_covid = np.sum(covid_by_tract, axis=0)

    cut = len(covid_timeline)
    if city == 'nola' or city == 'br':
        cut *= 7
        all_covid = np.divide(all_covid, 7)

    print(covid_timeline.shape)
    print(all_crime.shape, crime_timeline.shape)
    all_crime = all_crime[-cut:]
    crime_timeline = crime_timeline[-cut:]
    print(all_crime.shape, crime_timeline.shape)

    c4 = correlate(all_covid, all_crime)
    lags = correlation_lags(len(all_crime), len(all_covid))

    plot_c4(city, all_crime, crime_timeline, all_covid, covid_timeline, c4, lags)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('option', type=str, action='store',
                        help='generate or plot')
    parser.add_argument('city', type=str, action='store',
                        help='city code: sf, mk, or br')

    parser.add_argument('--covid_days', type=int, action='store', default=-1,
                        help='# of days from end of file to include as covid data')
    parser.add_argument('--val_months', type=int, action='store', default=7,
                        help='# of months before the beginning of covid to use as validation data')

    parser.add_argument('--test_covid', type=str, action='store', default='False',
                        help='if true, plot the covid testing data.')

    parser.add_argument('--m', type=int, action='store', default=1,
                        help='the period used for seasonal differencing')
    parser.add_argument('--month', type=int, action='store', default=30,
                        help='# of days in month')
    parser.add_argument('--year', type=int, action='store', default=365,
                        help='# of days in year')
    parser.add_argument('--poly', type=int, action='store', default=3,
                        help='order of polynomial used for savgol smoothing filter')

    parser.add_argument('--auto_save', type=str, action='store', default='False',
                        help='automatically save plots to drive instead of showing them.')

    parser.add_argument('--max_m', type=int, action='store', default=9,
                        help='maximum m for gen_best_params search')

    args = parser.parse_args()

    test_covid_ = str2bool(args.test_covid)
    smooth_together = False  # str2bool(args.smooth_together)
    auto_save_ = str2bool(args.auto_save)

    if args.city == 'all':
        cities = city_splits.keys()
    else:
        cities = [args.city]

    for city_ in cities:
        print('{}({})'.format(args.option.lower(), city_))
        if args.option.lower() == 'gen_data':   # generate city data
            gen_data(city_)
        elif args.option.lower() == 'plot_arima':    # plot data
            plot_arima(city_, city_splits[city_] if args.covid_days == -1 else args.covid_days, args.val_months,
                       test_covid=test_covid_, month=args.month, year=args.year, poly_order=args.poly, auto_save=auto_save_)
        elif args.option.lower() == 'gen_arima_params':  # find best model params
            params_ = generate_arima_params(city_, args.val_months, args.month, args.year, args.poly, max_m=args.max_m)
            print(params_)
        elif args.option.lower() == 'plot_cc':
            multi_plot_c4(city_)