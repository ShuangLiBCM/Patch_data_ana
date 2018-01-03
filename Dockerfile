FROM eywalker/tensorflow-jupyter:v1.0.0rc0

RUN pip3 install seaborn
RUN pip3 install -U scikit-learn
RUN pip3 install pandas_datareader
RUN pip3 install imblearn
RUN pip3 install xgboost
RUN pip3 install python-linkedin
RUN pip3 install --upgrade https://github.com/ozgur/python-linkedin/tarball/master
RUN pip3 install Theano
RUN pip3 install Keras
RUN pip3 install pillow
RUN pip3 install statsmodels

ADD . /src
RUN pip3 install -e /src