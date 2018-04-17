#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 22:48:27 2018

@author: mariomoreno
"""

import pandas as pd
import matplotlib.pyplot as plt

graffiti = 'graffiti.csv'
vacancies = 'vacancies.csv'
lights = 'lights.csv'

def summary_stats(graffiti, vacancies, lights):
    '''
    This function calculates summary statistics from our condensed dataframe
    '''
    
    c_db = read_csv(graffiti, vacancies, lights)
    
    counts(c_db)
    counts_per_month(c_db)
    counts_per_hood(c_db)
    vacant_buildings(c_db)
    time_to_solution(c_db)


def read_csv(path_a, path_b, path_c):
    '''
    This function reads in the three datasets, cleans them, and merges them
    '''
    
    grf = pd.read_csv(path_a)
    vac = pd.read_csv(path_b)
    lts = pd.read_csv(path_c)
    
    vac.rename(columns={"SERVICE REQUEST TYPE": "Type of Service Request", \
                        "X COORDINATE": "X Coordinate", \
                        "Y COORDINATE": "Y Coordinate", \
                        "ZIP CODE": "ZIP Code", \
                        "SERVICE REQUEST NUMBER": "Service Request Number",\
                        "LATITUDE": "Latitude", "LONGITUDE": "Longitude", \
                        "DATE SERVICE REQUEST WAS RECEIVED": "Creation Date",\
                        }, inplace = True)
    
    vac["Street Address"] = vac[vac.columns[10:14]].apply(lambda x: ' '.join(x.astype(str)), axis = 1)
    #vac.drop(vac.columns[3:14], axis = 1, inplace = True)
    
    condensed = grf.append([vac, lts])
    
    return condensed   
    
def counts(db):
    '''
    This function counts the type of services requested in 2017
    '''
    
    services = db.groupby(['Type of Service Request'])
    return services.size()

def counts_per_month(db):
    '''
    This function graphs the type of services requested per month in 2017.
    
    It does this by first transforming the dates in our database into datetime
    format, then using matplotlib to graph.
    '''
    
    services_per_month = db[["Type of Service Request", "Creation Date"]]
    services_per_month['Month'] = pd.DatetimeIndex(services_per_month['Creation Date']).month
    
    spms = services_per_month.groupby(['Type of Service Request', 'Month'])
    return spms.size()
    
def counts_per_hood(db):
    '''
    This function graphs the types of services requested per neighborhood
    in 2017.
    
    It looks at three indicators of neighborhood: ward, police district
    and community area.
    '''
    
    services_per_hood = db[['Type of Service Request', 'Community Area', \
                            'Police District', 'Ward']]
    
    comm_area = services_per_hood.groupby(['Type of Service Request', 'Community Area'])
    value_counts(comm_area, "Community Area")
        
        #if col == 'Police District':
         #   value_counts(services_per_hood, col)
        
        #if col == 'Ward':
         #   value_counts(services_per_hood, col)
            
def value_counts(db, column_name):
    '''
    Helper function that builds bar charts
    '''
    
    db[column_name].value_counts().plot(kind='barh')
    
    #services_per_hood['Police District'].value_counts().plot(kind='barh')
    #services_per_hood['Community Area'].value_counts().plot(kind='barh')
        
def vacant_buildings(db):
    '''
    This function provides details on the location of dangerous buildings
    by ward
    '''
    
    danger = db[["Type of Service Request", "IS THE BUILDING DANGEROUS OR HAZARDOUS?", \
                 "Ward"]]
    
    # table, dangerous buildings per ward
    
def time_to_solution(db):
    '''
    This function graphs how long it takes for 311 requests to be resolved
    per ward
    '''
    
    resolution = db[['Type of Service Request', 'Creation Date', 'Completion Date' \
                     'Ward']]
    
    # one bar chart: avg graffiti removal length per ward, avg lights fixed per ward length

    
    
    
    
    
    
    
    
    
    