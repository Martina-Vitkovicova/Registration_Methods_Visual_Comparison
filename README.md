The setup: 

In order to run the application, you need to have the registration_methods.py, application_dash.py,
application_html.py and constants.py scripts downloaded. <br>
The application is based on prostate cancer patients' data which is private,
which is why we can not share it along with the python scripts.
Although we add at least the computations_files directory, containing the precomputed data for the majority of graphs. 
Only the last section of the application - the timestamp section -- does not work without the private data. <br>
After downloading all the materials and placing them in one directory, 
the constant FILEPATH in the constants.py file needs to be changed according to the location of that directory. <br>
The last step of the setup is the import of necessary libraries: numpy, trimesh, pywavefront, scipy, plotly, dash, json and copy.
After that, you can start the application by running the application_dash.py script. <br>

Note: in the code documentation we use RM to denote registration methods
