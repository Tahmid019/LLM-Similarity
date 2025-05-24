import nltk
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from app.__init__ import logger

def download_nltk_data():
    logger.info("punkt >>>")
    nltk.download('punkt', quiet=True)
    logger.info("Downloaded.")

    logger.info("pubkt_tab >>>")
    nltk.download('punkt_tab', quiet=True)
    logger.info("Downloaded.")
    
    logger.info("wordnet >>>")
    nltk.download('wordnet', quiet=True)
    logger.info("Downloaded.")
    
    logger.info("omw-1.4 >>>")
    nltk.download('omw-1.4', quiet=True)
    logger.info("Downloaded.")
    
    return True
