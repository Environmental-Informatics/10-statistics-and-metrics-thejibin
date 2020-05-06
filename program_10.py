#!/bin/env python
# Created on March 25, 2020
#  by Keith Cherkauer
#
# This script servesa as the solution set for assignment-10 on descriptive
# statistics and environmental informatics.  See the assignment documention 
# and repository at:
# https://github.com/Environmental-Informatics/assignment-10.git for more
# details about the assignment.
#
import pandas as pd
import scipy.stats as stats
import numpy as np

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    ## Remove invalid streamflow data
    DataDF.Discharge[(DataDF['Discharge']<0)]=np.nan
    
    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    
    ## Clip the data to the data range
    DataDF=DataDF.loc[startDate:endDate]
    ## Find the number of missing values
    MissingValues=DataDF['Discharge'].isna().sum()
    
    return( DataDF, MissingValues )

def CalcTqmean(Qvalues):
    """This function computes the Tqmean of a series of data, typically
       a 1 year time series of streamflow, after filtering out NoData
       values.  Tqmean is the fraction of time that daily streamflow
       exceeds mean streamflow for each year. Tqmean is based on the
       duration rather than the volume of streamflow. The routine returns
       the Tqmean value for the given data array."""
    
    ## Filter out NoData
    data_q=Qvalues.dropna()
    ## Calculate Tqmean
    Tqmean=((data_q>data_q.mean()).sum()/len(data_q))
    return ( Tqmean )

def CalcRBindex(Qvalues):
    """This function computes the Richards-Baker Flashiness Index
       (R-B Index) of an array of values, typically a 1 year time
       series of streamflow, after filtering out the NoData values.
       The index is calculated by dividing the sum of the absolute
       values of day-to-day changes in daily discharge volumes
       (pathlength) by total discharge volumes for each year. The
       routine returns the RBindex value for the given data array."""
    
    ## Filter out NoData
    data_q=Qvalues.dropna()
    ## Calculate day-to-day dicahrge
    diff=data_q.diff()
    ## Calculate sum of absolute values of day-to-day discharge difference
    Total_abs=abs(diff).sum()
    ## Calculate total discharge volumes
    Total_discharge=Qvalues.sum()
    
    ## Estimate RBIndex
    RBindex=Total_abs/Total_discharge
    return ( RBindex )

def Calc7Q(Qvalues):
    """This function computes the seven day low flow of an array of 
       values, typically a 1 year time series of streamflow, after 
       filtering out the NoData values. The index is calculated by 
       computing a 7-day moving average for the annual dataset, and 
       picking the lowest average flow in any 7-day period during
       that year.  The routine returns the 7Q (7-day low flow) value
       for the given data array."""
      
    ## Filter out NoData
    data_q=Qvalues.dropna()
    ## Calculate the minimum value of 7-day average value
    val7Q=(data_q.rolling(7).mean()).min()
    
    return ( val7Q )

def CalcExceed3TimesMedian(Qvalues):
    """This function computes the number of days with flows greater 
       than 3 times the annual median flow. The index is calculated by 
       computing the median flow from the given dataset (or using the value
       provided) and then counting the number of days with flow greater than 
       3 times that value.   The routine returns the count of events greater 
       than 3 times the median annual flow value for the given data array."""
    
    ## Filter out NoData
    data_q=Qvalues.dropna()
    ## Find the number of days flow is greater than 3 times
    days=data_q>(data_q.median())*3
    ## Find the count of days
    median3x=days.sum()
    
    return ( median3x )

def GetAnnualStatistics(DataDF):
    """This function calculates annual descriptive statistcs and metrics for 
    the given streamflow time series.  Values are retuned as a dataframe of
    annual values for each water year.  Water year, as defined by the USGS,
    starts on October 1."""
    
    ## Use the contents in DF column as headers
    col_name=['site_no', 'Mean Flow', 'Peak Flow', 'Median Flow','Coeff Var','Skew','Tqmean',
                 'R-B Index', '7Q', '3xMedian']
    ## Create index and DataFrame
    data_yearly=DataDF.resample('AS-OCT').mean()
    WYDataDF=pd.DataFrame(index=data_yearly.index,columns=col_name)
    
    ## Calculate the statistics
    WYDataDF['site_no']=DataDF['site_no'].resample('AS-OCT').mean()
    WYDataDF['Mean Flow']=DataDF['Discharge'].resample('AS-OCT').mean()
    WYDataDF['Peak Flow']=DataDF['Discharge'].resample('AS-OCT').max()
    WYDataDF['Median Flow']=DataDF['Discharge'].resample('AS-OCT').median()
    WYDataDF['Coeff Var']=(DataDF['Discharge'].resample('AS-OCT').std()/
            DataDF['Discharge'].resample('AS-OCT').mean()*100)
    WYDataDF['Skew']=DataDF['Discharge'].resample('AS-OCT').apply(stats.skew)
    WYDataDF['Tqmean']=DataDF['Discharge'].resample('AS-OCT').apply(CalcTqmean)
    WYDataDF['R-B Index']=DataDF['Discharge'].resample('AS-OCT').apply(CalcRBindex)
    WYDataDF['7Q']=DataDF['Discharge'].resample('AS-OCT').apply(Calc7Q)
    WYDataDF['3xMedian']=DataDF['Discharge'].resample('AS-OCT').apply(CalcExceed3TimesMedian)
     
    return ( WYDataDF )

def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""
    
    ## Create column name
    col_name=['site_no', 'Mean Flow', 'Coeff Var','Tqmean','R-B Index']
    
    ## Create index and DataFrame
    data_monthly=DataDF.resample('MS').mean()
    MoDataDF=pd.DataFrame(index=data_monthly.index,columns=col_name)
    
    MoDataDF['site_no']=DataDF['site_no'].resample('MS').mean()
    MoDataDF['Mean Flow']=DataDF['Discharge'].resample('MS').mean()
    MoDataDF['Coeff Var']=(DataDF['Discharge'].resample('MS').std()/
            DataDF['Discharge'].resample('MS').mean()*100)
    MoDataDF['Tqmean']=DataDF['Discharge'].resample('MS').apply(CalcTqmean)
    MoDataDF['R-B Index']=DataDF['Discharge'].resample('MS').apply(CalcRBindex)
    
    return ( MoDataDF )

def GetAnnualAverages(WYDataDF):
    """This function calculates annual average values for all statistics and
    metrics.  The routine returns an array of mean values for each metric
    in the original dataframe."""
    
    ## Generate annual averages from Water Year Data
    AnnualAverages=WYDataDF.mean(axis=0)
    return( AnnualAverages )

def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
    
    ## Generate Monthly Averages from Monthly Data
    MonthlyAverages=MoDataDF.groupby(by=[MoDataDF.index.month]).mean()
    return( MonthlyAverages )

# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define filenames as a dictionary
    # NOTE - you could include more than jsut the filename in a dictionary, 
    #  such as full name of the river or gaging site, units, etc. that would
    #  be used later in the program, like when plotting the data.
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    MissingValues = {}
    WYDataDF = {}
    MoDataDF = {}
    AnnualAverages = {}
    MonthlyAverages = {}
    
    # process input datasets
    for file in fileName.keys():
        
        print( "\n", "="*50, "\n  Working on {} \n".format(file), "="*50, "\n" )
        
        DataDF[file], MissingValues[file] = ReadData(fileName[file])
        print( "-"*50, "\n\nRaw data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # clip to consistent period
        DataDF[file], MissingValues[file] = ClipData( DataDF[file], '1969-10-01', '2019-09-30' )
        print( "-"*50, "\n\nSelected period data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # calculate descriptive statistics for each water year
        WYDataDF[file] = GetAnnualStatistics(DataDF[file])
        
        # calcualte the annual average for each stistic or metric
        AnnualAverages[file] = GetAnnualAverages(WYDataDF[file])
        
        print("-"*50, "\n\nSummary of water year metrics...\n\n", WYDataDF[file].describe(), "\n\nAnnual water year averages...\n\n", AnnualAverages[file])

        # calculate descriptive statistics for each month
        MoDataDF[file] = GetMonthlyStatistics(DataDF[file])

        # calculate the annual averages for each statistics on a monthly basis
        MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
        
        print("-"*50, "\n\nSummary of monthly metrics...\n\n", MoDataDF[file].describe(), "\n\nAnnual Monthly Averages...\n\n", MonthlyAverages[file])
        
        ## Add station name using the key name as last column to both Annual and Monthly Data
        WYDataDF[file]['Station']=file
        AnnualAverages[file]['Station']=file
        MoDataDF[file]['Station']=file
        MonthlyAverages[file]['Station']=file
        
    ## Output annual and monthly metrics as CSV file
    ## Use concat so that each file contain data for both rivers
    annual_metrics=pd.concat([WYDataDF['Wildcat'],WYDataDF['Tippe']],axis=0)
    monthly_metrics=pd.concat([MoDataDF['Wildcat'],MoDataDF['Tippe']],axis=0)
    annual_metrics.to_csv('Annual_Metrics.csv',sep=',')
    monthly_metrics.to_csv('Monthly_Metrics.csv',sep=',')
    
    ## Output annual and monthly averages to TAB delimited files
    ## Use concat so that each file contain data for both rivers
    annual_average_metrics=pd.concat([AnnualAverages['Wildcat'],AnnualAverages['Tippe']],axis=0)
    monthly_average_metrics=pd.concat([MonthlyAverages['Wildcat'],MonthlyAverages['Tippe']],axis=0)
    annual_average_metrics.to_csv('Average_Annual_Metrics.csv',sep='\t')
    monthly_average_metrics.to_csv('Average_Monthly_Metrics.csv',sep='\t')
    
    