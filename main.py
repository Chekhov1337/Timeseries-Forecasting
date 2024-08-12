from modules.models import *
from modules.preprocessing import *
from modules.stats import *

values = read_tsf()
x_train, x_test = split_data(values)

arma = arma_model(x_train)
plot(x_train, arma, x_test, 'ARMA')
