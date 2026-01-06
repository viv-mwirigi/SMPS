Variables stored in separate files (Header+values)

Filename

	Data_separate_files_header_startdate(YYYYMMDD)_enddate(YYYYMMDD)_userid_randomstring_currrentdate(YYYYMMDD).zip
	
	e.g., Data_separate_files_header_20050316_20050601.zip

	
Folder structure

	Networkname
		Stationname

		
Dataset Filename

	CSE_Network_Station_Variablename_depthfrom_depthto_startdate_enddate.ext

	CSE	- Continental Scale Experiment (CSE) acronym, if not applicable use Networkname
	Network	- Network abbreviation (e.g., OZNET)
	Station	- Station name (e.g., Widgiewa)
	Variablename - Name of the variable in the file (e.g., Soil-Moisture)
	depthfrom - Depth in the ground in which the variable was observed (upper boundary)
	depthto	- Depth in the ground in which the variable was observed (lower boundary)
	startdate -	Date of the first dataset in the file (format YYYYMMDD)
	enddate	- Date of the last dataset in the file (format YYYYMMDD)
	ext	- Extension .stm (Soil Temperature and Soil Moisture Data Set see CEOP standard)
	
	e.g., OZNET_OZNET_Widgiewa_Soil-Temperature_0.150000_0.150000_20010103_20090812.stm

	
File Content Sample
	
	REMEDHUS   REMEDHUS        Zamarron          41.24100    -5.54300  855.00    0.05    0.05  (Header)
	2005/03/16 00:00    10.30 U	M	(Records)
	2005/03/16 01:00     9.80 U M

	
Header

	CSE Identifier - Continental Scale Experiment (CSE) acronym, if not applicable use Networkname
	Network	- Network abbreviation (e.g., OZNET)
	Station	- Station name (e.g., Widgiewa)
	Latitude - Decimal degrees. South is negative.
	Longitude - Decimal degrees. West is negative.
	Elevation - Meters above sea level
	Depth from - Depth in the ground in which the variable was observed (upper boundary)
	Depth to - Depth in the ground in which the variable was observed (lower boundary)

	
Record

	UTC Actual Date and Time
	yyyy/mm/dd HH:MM
	Variable Value
	ISMN Quality Flag
	Data Provider Quality Flag, if existing


Network Information

	TAHMO
		Abstract: The Trans-African HydroMeteorological Observatory (TAHMO) aims to develop a vast network of weather stations across Africa. Current and historic weather data is important for agricultural, climate monitoring, and many hydro-meteorological applications.
		Continent: Africa
		Country: Côte d'Ivoire, Nigeria, Ghana, Uganda, Rwanda, Kenya
		Stations: 70
		Status: running
		Data Range: from 2015-06-17 
		Type: project
		Url: https://tahmo.org/
		Reference: We acknowledge the work of Frank Annor and Nicolaas Cornelis van de Giesen and the Trans-African Hydro-Meterological Observatory (TAHMO) network community in support of the ISMN;
		Variables: precipitation, soil temperature, air temperature, soil moisture, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.30 - 0.30 m, 0.60 - 0.60 m, 2.00 - 2.00 m
		Soil Moisture Sensors: TEROS12, GS1, TEROS10, 

