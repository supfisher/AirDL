import logging
logging.basicConfig(filename='../toy_examples/.logging/logger.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Simulator')