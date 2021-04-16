from GeoSeries import *
import os

sep = os.path.sep
data_p = '..' + sep + 'data'

if __name__ == '__main__':
    crimeData = CrimeSeries(sep.join([data_p, 'SF_crime.csv']))
    covidData = CovidSeries(sep.join([data_p, 'sf_covid.csv']))

    fmap = crimeData.get_fmap('date', filterBy='type', filters=['LARCENY THEFT'])
    GeoSeries.print_fmap(fmap)
    plot_data(fmap, ylabel="occurences", title="2020 Crime in San Francisco")

    fmap = crimeData.get_fmap('type')
    GeoSeries.print_fmap(fmap)
    plot_data(fmap, ylabel="#", xlabel='type', title="2020 Crime in San Francisco", xticks=len(fmap.T[0]))

    fmap = covidData.get_fmap()
    GeoSeries.print_fmap(fmap)
    plot_data(fmap, ylabel="new cases", title="New Daily COVID Cases in San Francisco")