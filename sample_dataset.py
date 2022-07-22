#!C:\Python\Python36\python.exe

'''

    File: sample_dataset.py
    Author: Robert Church robert@robertchurch.us
    Date: 2022 July 07
    Course: ADTA 5340
    Assignment: Final Project
    
    Purpose:  Purpose of this script is to sample the first 250,000
              entries of the file flows.txt add a header row, then
              write to a sample file.
              
              Note: There are better nad more robust ways to do this,
              but the task is simple so why add complexity to the
              Python code.

    Input:    The file flows.txt was obtained from the file flows.tar.gz
              obtained here: https://csr.lanl.gov/data/cyber1/

    Output:   The output file flow_sample.csv will contain a sample of the
              entries contained within flows.txt and will be utilized as
              input for the Final Project algorithm training model.

'''

number_of_samples = 250000
input_datset_file="flows.txt"
output_sample_file="flow_sample.csv"

'''
    Pseudo Code:

        Open flow.txt
        Append first number_of_samples into list
        Add header row to first entry in list
        Write list to sample file
        Close sample file

'''

with open(input_datset_file, "r") as input_dataset:
    sample_data = [next(input_dataset) for x in range(number_of_samples)]

sample_data.insert(0, "time,duration,source computer,source port,destination computer,destination port,protocol,packet count,byte count\n")

with open(output_sample_file, "w") as output_sample:
    output_sample.write(''.join(sample_data))

output_sample.close()


