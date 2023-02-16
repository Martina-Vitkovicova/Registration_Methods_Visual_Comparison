The setup and information: 

In order to run the application, you need to have the **registration_methods.py**, **application_dash.py**,
**application_html.py**, **constants.py** scripts and **assets** directory downloaded. <br>
The application is based on private prostate cancer patients' data*,
which is why we can not share it along with the other files.
Although we add at least the **computations_files** directory, containing the precomputed data for the majority of graphs. 
Only the last section of the application - the timestamp section -- does not work without the private data. <br>
After downloading all the materials and placing them in one directory, 
the constant **FILEPATH** in the constants.py file needs to be changed according to the location of that directory. <br>
The last step of the setup is the import of necessary libraries: **numpy, trimesh, pywavefront, scipy, plotly, dash, json and copy**.
After that, you can start the application by *running the application_dash.py* script. <br>


*The data is structured in a way that every patient has their own directory named by their ID. <br>
In that directory, there are four directories containing the patient's anatomy: bones, prostate, bladder and rectum. <br>
The directories are composed of 13 meshes of the particular organ/bone in different timestamps and one plan mesh, all in .obj file format. 


**Note**: We use RM to denote registration methods within the code documentation.
