import argparse
import csv
import httplib2
import os
import simplejson

from analysis_engine.model_information import (
    get_aileron_map,
    get_conf_map,
    get_flap_map,
    get_slat_map,

)
from analysis_engine.velocity_speed import get_vspeed_map


# Switch to production when it is updated to include series api.
BASE_URL = 'https://polaris.flightdataservices.com' # 'https://polaris.flightdataservices.com'

def get_series():
    '''
    Looks up currently active Aircraft Series and Families via web API and
    checks each mapping for entries
    '''
    http = httplib2.Http(disable_ssl_certificate_validation=True)
    try:
        response, content = http.request(BASE_URL + '/api/aircraft/active_series.json')
    except Exception as err:
        print 'Exception raised when querying website:', str(err)
        raise
    else:
        series_list = simplejson.loads(content)['series']
    lookups = (get_aileron_map, get_conf_map, get_flap_map, get_slat_map, get_vspeed_map)
    for series_info in series_list:
        series = series_info['Series']
        family = series_info['Family']
        for lookup in lookups:
            lookup_name = lookup.__name__.split('_')[1]
            try:
                series_info[lookup_name] = 'Y' if lookup(series=series, family=family) else 'N/R'
            except KeyError:
                series_info[lookup_name] = 'N'
    return series_list

def save_series_coverage(dest_path):
    series_info = get_series()
    with open(dest_path, 'wb') as _file:
        fieldnames = ['Series', 'Family', 'aileron', 'flap',  'slat', 'conf', 'vspeed']
        dw = csv.DictWriter(_file, delimiter='\t', fieldnames=fieldnames)
        dw.writerow(dict((n,n.title()) for n in fieldnames))
        for row in series_info:
            dw.writerow(row)


def create_parser():
    parser = argparse.ArgumentParser(description='Check for existing mappings'\
                        'for currently active Aircraft series and families.')
    parser.add_argument('-r', '--report_path', default=None, 
        help='Specify the export location of generated report,' \
        'file name will be "series_mapping_report.csv"')
    parser.add_argument('-b', '--base_url', default=None, 
        help='Base URL of location of API to use')
    return parser


if __name__=='__main__':
    # Parse command line arguments
    file_name = 'series_mapping_report.csv'
    parser = create_parser()
    args = parser.parse_args()

    if args.report_path and os.path.isdir(args.report_path):
        report_path = os.path.join(args.report_path, file_name)
    else:
        report_path = file_name

    if args.base_url:
        BASE_URL = args.base_url

    save_series_coverage(report_path)
