FROM eywalker/tensorflow-jupyter:v1.0.0rc0

RUN pip3 install seaborn
RUN pip3 install -U scikit-learn
RUN pip3 install pandas_datareader
RUN pip3 install pillow
RUN pip3 install statsmodels
RUN apt-get update
RUN apt-get install stimfit python-stfio -y
WORKDIR /src

ADD . /src/Patch_ana
RUN pip3 install -e /src/Patch_ana
