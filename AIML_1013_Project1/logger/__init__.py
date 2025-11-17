import logging 
import os 
from from_root import from_root
from datetime import datetime 

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_dir = 'logs'

log_path = os.path.join(from_root(), log_dir, LOG_FILE)

os.makedirs(log_dir, exist_ok=True)


#Logging Configs 

logging.basicConfig(
    filename = log_path,
    format = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    level = logging.DEBUG
)