from modules.models import *
from modules.preprocessing import *
from modules.stats import *

values = read_tsf()
x_train, x_test = split_data(values)

holt = holt_winters(x_train)
metrics, avg = metrics(holt, x_test, True)
