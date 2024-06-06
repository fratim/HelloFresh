
import datetime

SEED = 0
DEBUG =  True
DEBUG_SAMPLES = 5

# convert target dates to datstime objects
TARGET_START_DATE_WIKI = datetime.datetime(2024, 2, 15)
TARGET_END_DATE_WIKI = datetime.datetime(2024, 3, 15)

TARGET_START_DATE_X = datetime.datetime(2024, 3, 1)
TARGET_END_DATE_X = datetime.datetime(2024, 4, 1)

N_SAMPLES_PER_MONTH_PER_CAT = 250

MAX_SAMPLE_INFO_LENGTH_CHARS = 4000

CHUNK_SIZE = 100
TEST_CHUNK_SIZE = 12

WIKI_CHUNKS = 27

if DEBUG:
    N_ARTICLES_LIMIT = 2 
else:
    N_ARTICLES_LIMIT = 5 # Do not update this without updating trhe text prompts

