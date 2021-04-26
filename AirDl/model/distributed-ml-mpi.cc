#include "distributed-ml-mpi.h"
#include "ns3/string.h"
#include <dlfcn.h>
#include "ns3/mpi-module.h"
#include "ns3/core-module.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DistributedMlTcpMpi");
// NS_OBJECT_ENSURE_REGISTERED (DistributedMlTcpMpi);


void MpiHelper::Enable(std::vector<char*> argv, bool nullmsg){
	//std::cout << "argv: " << argv[0] << std::endl;
	int argc = static_cast<int>(argv.size());
	//std::cout << "argc: " << argc << std::endl;

	char** carg = new char*();
	for(int i=0; i<argc; i++){
		//std::cout << "debug point 0... " <<std::endl;
		*(carg+i) = argv[i];
		//std::cout << "debug point -1... " <<std::endl;
	}

	//std::cout << "carg: " << *carg << std::endl;

	if(nullmsg)
    {
      GlobalValue::Bind ("SimulatorImplementationType",
                         StringValue ("ns3::NullMessageSimulatorImpl"));
    }
  	else
    {
      GlobalValue::Bind ("SimulatorImplementationType",
                         StringValue ("ns3::DistributedSimulatorImpl"));
    }
    //std::cout << "debug point 1: " << std::endl;
	dlopen("libmpi.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);

	//std::cout << "debug point 2: " << std::endl;
	MpiInterface::Enable (&argc, &carg);
	//std::cout << "debug point 3: " << std::endl;
}


// void MpiHelper::Disable(){
//   NS_ASSERT (g_parallelCommunicationInterface);
//   g_parallelCommunicationInterface->Disable ();
//   delete g_parallelCommunicationInterface;
//   g_parallelCommunicationInterface = 0;

// }

}