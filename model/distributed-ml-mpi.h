#ifndef DISTRIBUTED_ML_MPI_H
#define DISTRIBUTED_ML_MPI_H

#include "ns3/command-line.h"
#include "ns3/mpi-interface.h"
#include <vector>
#include <string>
#include <iostream>

namespace ns3 {


class MpiHelper: MpiInterface{
public:
	// void Enable(int argc, std::string arg);

	void Enable(std::vector<char*> argv, bool nullmsg=false);
};



}


#endif