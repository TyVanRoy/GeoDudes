import csv
import argparse
import numpy as np
from matplotlib import pyplot as plt 
from pandas import DataFrame
from tqdm import tqdm

col_index = {
    'date': 1,
    'loc': 0,
    'case': 2
}

alt_col_index = {
    'date': 3,
    'loc': 0,
    'case': 4
}

class GeoSeries(object):

    def __init__(self, filename):
        self.filename = filename
        self.names, self.row_data = self.__load_data()

        self.col_dict = dict.fromkeys(self.names, [])

        print("Constructing dictionary...")
        for i, col in enumerate(self.row_data.T):
            self.col_dict[self.names[i]] = col

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.col_dict[key.lower()]
        else:
            return self.col_dict[self.names[key]]

    def __load_data(self):
        with open(self.filename, newline='') as csvfile:
            data = []
            col_names = []
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i, row in enumerate(spamreader):
                if i == 0:
                    col_names = [r.lower() for r in row]
                    col_names[0] = col_names[0].replace('\ufeff', '')
                else:
                    row = self.clean(row)
                    if row != 0:
                        data.append(row)
                

            print("Data loaded from {}. Column names: {}".format(self.filename, col_names))
            return col_names, np.asarray(data)

    def toDataFrame(self):
        return DataFrame.from_dict(self.col_dict)

    @staticmethod
    def scrap_year(row):
        return ['/'.join(row[0].split('/')[:-1]), row[1], row[2]]

    @staticmethod
    def print_fmap(fmap, longest_string=50):
        for value in fmap:
            print(("{:-<%d}{:^10}" % longest_string).format(*value))


class CrimeSeries(GeoSeries):

    def get_fmap(self, key, filterBy='', filters=None, loadbar=True):
        key = key.lower()
        filterBy = filterBy.lower()
        if key == filterBy:
            print("Key == filterBy, skipping filtering...")
        if filterBy != '' and filters != None:
            return self.construct_fmap(self[key], self[filterBy], filters, loadbar=loadbar)
        return self.construct_fmap(self[key], loadbar=loadbar)

    @staticmethod
    def clean(row):
        row = CrimeSeries.merge_crime(row)
        # row = GeoSeries.scrap_year(row)
        # row = CrimeSeries.filterYear(row, ['2018', '2019', '2020', '2021'])
        return row

    @staticmethod
    def merge_crime(row):
        if len(row) > 3:
            return [row[0], ' '.join(row[1:-1]), row[-1]]
        return row


    def filterYear(row, years):
        inrow = False
        for y in years:
            inrow = y in row[0]
            if inrow:
                break
        if inrow:
            return row
        else:
            return 0

    @staticmethod
    def construct_fmap(data, filterData=[], filters=None, loadbar=True):
        data_set = sorted(set(data), key=np.asarray(data).tolist().index)
        fmap = []
        if len(filterData) == 0:
            if loadbar:
                print("Constructing fmap...")
            for value in tqdm(data_set) if loadbar else data_set:
                fmap.append([value, data.tolist().count(value)])
        else:
            filters = [str(f).lower() for f in filters]
            row_counter = 0
            if loadbar:
                print("Initializing fmap...")
            for d in tqdm(data_set) if loadbar else data_set:
                fmap.append([d, 0])
            if loadbar:
                print("Constructing fmap...")
            for i, element in tqdm(enumerate(filterData)) if loadbar else enumerate(filterData):
                if data[i] != data_set[row_counter]:
                    row_counter += 1
                
                fmap[row_counter][1] += 1 if element.lower() in filters else 0

        return np.asarray(fmap)


class CovidSeries(GeoSeries):

    def get_fmap(self, locFilters=[], loadbar=True, alt_index=False):
        if alt_index:
            return self.construct_fmap(self[alt_col_index['date']], self[alt_col_index['case']], self[alt_col_index['loc']],
                                       locFilters=locFilters, loadbar=loadbar)
        return self.construct_fmap(self[col_index['date']], self[col_index['case']], self[col_index['loc']], locFilters=locFilters, loadbar=loadbar)

    @staticmethod
    def clean(row):
        # row = GeoSeries.scrap_year(row)
        return row

    @staticmethod
    def construct_fmap(dates, counts, locs, locFilters=[], loadbar=True):
        fmap = []

        date_set = sorted(set(dates), key=dates.tolist().index)
        date_counter = 0

        if loadbar:
            print("Initializing fmap...")
        for date in tqdm(date_set) if loadbar else date_set:
            fmap.append([date, 0])

        if loadbar:
            print("Constructing fmap...")
        for i, count in tqdm(enumerate(counts)) if loadbar else enumerate(counts):
            if dates[i] != date_set[date_counter]:
                date_counter += 1
            
            if len(locFilters) > 0:
                fmap[date_counter][1] += int(counts[i]) if locs[i] in locFilters else 0
            else:
                fmap[date_counter][1] += int(counts[i])

        return np.asarray(fmap)


def plot_data(fmap, xlabel='date', ylabel='#', title='Title', xticks=12, yticks=10):
    fmapT = fmap.T.tolist()
    fmapT[1] = [int(fmapT[1][i]) for i in range(len(fmapT[1]))]

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(fmapT[1])

    xaxis = np.linspace(0, len(fmapT[0]) - 1, xticks, dtype=int)
    plt.xticks(xaxis, fmap.T[0][xaxis])

    yaxis = np.linspace(0, max(fmapT[1]) + 10, yticks, dtype=int)
    plt.yticks(yaxis)

    plt.gcf().autofmt_xdate()
    plt.show()