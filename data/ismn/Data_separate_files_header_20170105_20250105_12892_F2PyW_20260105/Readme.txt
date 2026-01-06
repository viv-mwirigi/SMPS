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

	AMMA-CATCH
		Abstract: This network consists of three supersites in Benin, Niger and Mali. Mali works operational since 2005, Niger and Benin since 2006. Several measurements in Mali and Niger are taken at the same station with the same sensor type in the same depth. They are located at the bottom (sensor CS616_1), middle (CS616_2) or top (CS616_3) of dune slopes (Mali) or from a plateau to the valley bottom (Niger).
		Continent: Africa
		Country: Benin, Niger, Mali
		Stations: 7
		Status: running
		Data Range: from 2005-01-01 
		Type: project
		Url: http://www.amma-catch.org
		Reference: Mougin, E., Hiernaux, P., Kergoat, L., Grippa, M., de Rosnay, P., Timouk, F., Le Dantec, V., Demarez, V., Lavenu, F., Arjounin, M., Lebel, T. et al., 2009. The AMMA-CATCH Gourma observatory site in Mali: Relating climatic variations to changes in vegetation, surface in press, hydrology, fluxes and natural resources. Journal of Hydrology, 375(1-2): 14-33, https://doi.org/10.1016/j.jhydrol.2009.06.045;

Cappelaere, C., Descroix, L., Lebel, T., Boulain, N., Ramier, D., Laurent, J.-P., Le Breton, E., Boubkraoui, S., Bouzou Moussa, I. et al., 2009. The AMMA Catch observing system in the cultivated Sahel of South West Niger- Strategy, Implementation and Site conditions, 2009. Journal of Hydrology, 375(1-2): 34-51, https://doi.org/10.1016/j.jhydrol.2009.06.021;

de Rosnay, P., Gruhier, C., Timouk, F., Baup, F., Mougin, E., Hiernaux, P., Kergoat, L., and LeDantec, V.: Multi-scale soil moisture measurements at the Gourma meso-scale site in Mali, Journal of Hydrology, 375, 241-252, 2009, https://doi.org/10.1016/j.jhydrol.2009.01.015;

Lebel, Thierry, Cappelaere, Bernard, Galle, Sylvie, Hanan, Niall, Kergoat, Laurent, Levis, Samuel, Vieux, Baxter, Descroix, Luc, Gosset, Marielle, Mougin, Eric, Peugeot, Christophe and Seguis, Luc, 2009: AMMA-CATCH studies in the Sahelian region of West-Africa: An overview. JOURNAL OF HYDROLOGY, 375, 3-13, https://doi.org/10.1016/j.jhydrol.2009.03.020;

Galle, S., Grippa, M., Peugeot, C., Bouzou Moussa, I., Cappelaere, B., Demarty, J., Mougin, E., Lebel, T. and Chaffard, V., 2015, December. AMMA-CATCH a Hydrological, Meteorological and Ecological Long Term Observatory on West Africa: Some Recent Results. In AGU Fall Meeting Abstracts (Vol. 2015, pp. GC42A-01).;
		Variables: soil moisture, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.10 - 0.40 m, 0.20 - 0.20 m, 0.40 - 0.40 m, 0.40 - 0.70 m, 0.60 - 0.60 m, 0.70 - 1.00 m, 1.00 - 1.00 m, 1.00 - 1.30 m, 1.05 - 1.35 m, 1.20 - 1.20 m
		Soil Moisture Sensors: CS616, 

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

