Cab mobility traces are provided by the Exploratorium - the museum of science, art and human perception through the cabspotting project: http://cabspotting.org . "Each San Francisco based Yellow Cab vehicle is currently outfitted with a GPS tracking device that is used by dispatchers to efficiently reach customers. The data is transmitted from each cab to a central receiving station, and then delivered in real-time to dispatch computers via a central server. This system broadcasts the cab call number, location and whether the cab currently has a fare."(*) You can collect your own cab mobility traces following the instructions from http://cabspotting.org/api . 
You can also use this data set of cab mobility traces that were collected in May 2008. This archive contains file '_cabs.txt' with the list of all cabs and for each cab its mobility trace in a separate ASCII file, e.g. 'new_abboip.txt'. The format of each mobility trace file is the following - each line contains [latitude, longitude, occupancy, time], e.g.: [37.75134 -122.39488 0 1213084687], where latitude and longitude are in decimal degrees, occupancy shows if a cab has a fare (1 = occupied, 0 = free) and time is in UNIX epoch format.

If you intend to publish your results based on this data set, we appreciate if you cite the following publication in which we extensively use this data set:
@InProceedings{comsnets09piorkowski,
   title		= "{A Parsimonious Model of Mobile Partitioned Networks with Clustering}",
   author		= "Michal Piorkowski and Natasa Sarafijanovoc-Djukic and Matthias Grossglauser",
   booktitle	= "The First International Conference on COMmunication Systems and NETworkS (COMSNETS)",
   location		= "Bangalore, India",
   year			= "2009",
   month		= "January",
   url			= "http://www.comsnets.org",
   details		= "http://infoscience.epfl.ch/record/130383"
}

(*) http://cabspotting.org/about.html