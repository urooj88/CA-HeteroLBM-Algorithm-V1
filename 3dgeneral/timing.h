#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <sys/time.h>

/// Gets the current time in seconds

double get_wall_time(){
	
    struct timeval time;
    
    if (gettimeofday(&time,NULL)){
		
        //  Handle error
        return 0;
        
    }
    
    return (double)time.tv_sec + (double)time.tv_usec * 0.000001;
    
}
