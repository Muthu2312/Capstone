{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "71e7aacc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "        <script type=\"text/javascript\">\n",
       "        window.PlotlyConfig = {MathJaxConfig: 'local'};\n",
       "        if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: \"STIX-Web\"}});}\n",
       "        if (typeof require !== 'undefined') {\n",
       "        require.undef(\"plotly\");\n",
       "        requirejs.config({\n",
       "            paths: {\n",
       "                'plotly': ['https://cdn.plot.ly/plotly-2.12.1.min']\n",
       "            }\n",
       "        });\n",
       "        require(['plotly'], function(Plotly) {\n",
       "            window._Plotly = Plotly;\n",
       "        });\n",
       "        }\n",
       "        </script>\n",
       "        "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'lightgbm'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Temp\\ipykernel_25940\\2912334576.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m     18\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mensemble\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mAdaBoostClassifier\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     19\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0msvm\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 20\u001b[1;33m \u001b[1;32mimport\u001b[0m \u001b[0mlightgbm\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mlgb\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     21\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0mlightgbm\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mLGBMClassifier\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     22\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mxgboost\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mxgb\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'lightgbm'"
     ]
    }
   ],
   "source": [
    "import pandas as pd \n",
    "import numpy as np\n",
    "import matplotlib\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "%matplotlib inline \n",
    "import plotly.graph_objs as go\n",
    "import plotly.figure_factory as ff\n",
    "from plotly import tools\n",
    "from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot\n",
    "init_notebook_mode(connected=True)\n",
    "import gc\n",
    "from datetime import datetime \n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.model_selection import KFold\n",
    "from sklearn.metrics import roc_auc_score\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.ensemble import AdaBoostClassifier\n",
    "from sklearn import svm\n",
    "import lightgbm as lgb\n",
    "from lightgbm import LGBMClassifier\n",
    "import xgboost as xgb\n",
    "pd.set_option('display.max_columns', 100)\n",
    "\n",
    "\n",
    "RFC_METRIC = 'gini'  #metric used for RandomForrestClassifier\n",
    "NUM_ESTIMATORS = 100 #number of estimators used for RandomForrestClassifier\n",
    "NO_JOBS = 4 #number of parallel jobs used for RandomForrestClassifier\n",
    "\n",
    "\n",
    "#TRAIN/VALIDATION/TEST SPLIT\n",
    "#VALIDATION\n",
    "VALID_SIZE = 0.20 # simple validation using train_test_split\n",
    "TEST_SIZE = 0.20 # test size using_train_test_split\n",
    "\n",
    "#CROSS-VALIDATION\n",
    "NUMBER_KFOLDS = 5 #number of KFolds for cross-validation\n",
    "\n",
    "\n",
    "\n",
    "RANDOM_STATE = 2018\n",
    "MAX_ROUNDS = 1000 #lgb iterations\n",
    "EARLY_STOP = 50 #lgb early stop \n",
    "OPT_ROUNDS = 1000  #To be adjusted based on best validation rounds\n",
    "VERBOSE_EVAL = 50 #Print out metric result\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0ddab9c7",
   "metadata": {},
   "source": [
    "# data understanding\n",
    "About this file\n",
    "Training set for Credit Card Transactions\n",
    "1.index - Unique Identifier for each row\n",
    "2.transdatetrans_time - Transaction DateTime\n",
    "3.cc_num - Credit Card Number of Customer\n",
    "4.merchant - Merchant Name\n",
    "5.category - Category of Merchant\n",
    "6.amt - Amount of Transaction\n",
    "7.first - First Name of Credit Card Holder\n",
    "8.last - Last Name of Credit Card Holder\n",
    "9.gender - Gender of Credit Card Holder\n",
    "10.street - Street Address of Credit Card Holder\n",
    "11.city - City of Credit Card Holder\n",
    "12.state - State of Credit Card Holder\n",
    "13.zip - Zip of Credit Card Holder\n",
    "14.lat - Latitude Location of Credit Card Holder\n",
    "15.long - Longitude Location of Credit Card Holder\n",
    "16.city_pop - Credit Card Holder's City Population\n",
    "17.job - Job of Credit Card Holder\n",
    "18.dob - Date of Birth of Credit Card Holder\n",
    "19.trans_num - Transaction Number\n",
    "20.unix_time - UNIX Time of transaction\n",
    "21.merch_lat - Latitude Location of Merchant\n",
    "22.merch_long - Longitude Location of Merchant\n",
    "23.is_fraud - Fraud Flag <--- Target Class"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "20fe30b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.read_csv('fraudTest.csv',index_col=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "42256689",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7b26be56",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b3f00efc",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "24bfbcfd",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['is_fraud'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0133a852",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1=pd.read_csv('fraudTrain.csv',index_col=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "398fb3fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9fb360ee",
   "metadata": {},
   "source": [
    "# data dictionary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e90ec480",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4f07db35",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "25270cd5",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e64777fc",
   "metadata": {},
   "source": [
    "# Variable Categorization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "923f1e82",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1_num=df1.select_dtypes(include=[np.number])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f2ce267c",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1_num"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "52c1c37b",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1_cat=df1.select_dtypes(exclude=[np.number])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b0d4ea90",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1_cat"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "375cf2f9",
   "metadata": {},
   "source": [
    "# count of missing/ null values, redundant columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b9d17199",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4d6bb785",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1.head(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fd37be49",
   "metadata": {},
   "source": [
    "# Data Exploration"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a26a732a",
   "metadata": {},
   "source": [
    "# relationship between variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8f9a99a1",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1.corr()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1e4e1179",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.rcParams['figure.figsize']=(20,20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "252b3024",
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.heatmap(df1.corr(),annot=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ca32dede",
   "metadata": {},
   "source": [
    "# 1.Multi collinearity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a0ffac10",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1_n=df1_num.drop(['cc_num','is_fraud'],axis=1)\n",
    "df1_n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cb47a11a",
   "metadata": {},
   "outputs": [],
   "source": [
    "from statsmodels.stats.outliers_influence import variance_inflation_factor\n",
    "X=df1_n\n",
    "vif = pd.DataFrame()\n",
    "vif[\"VIF_Factor\"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]\n",
    "vif[\"Features\"] = X.columns\n",
    "print(vif)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cc3fa382",
   "metadata": {},
   "outputs": [],
   "source": [
    "vif.sort_values('VIF_Factor',ascending=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d25b01b1",
   "metadata": {},
   "source": [
    "# 2. Distribution of variables\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5023b5df",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1_num.drop(['cc_num','is_fraud'],axis=1).hist()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3a35f780",
   "metadata": {},
   "source": [
    "# 3.Presence of outliers and its treatment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "62a7907d",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1_n.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bcf16a56",
   "metadata": {},
   "outputs": [],
   "source": [
    "heart_CAT = ['amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat',\n",
    "       'merch_long']\n",
    "\n",
    "a = 2  \n",
    "b = 4  \n",
    "c = 1  \n",
    "\n",
    "fig = plt.figure(figsize=(14,10))\n",
    "\n",
    "for i in heart_CAT:\n",
    "    plt.subplot(a, b, c)\n",
    "    plt.title('{}, subplot: {}{}{}'.format(i, a, b, c))\n",
    "    plt.xlabel(i)\n",
    "    sns.boxplot(df1_n[i],color='red')\n",
    "    c = c + 1\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8e1bec03",
   "metadata": {},
   "source": [
    "# univariate analysis for numerical columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dce00bd8",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = plt.figure(figsize=(10,4))\n",
    "\n",
    "#  subplot #1\n",
    "plt.subplot(121)\n",
    "plt.title('credit card holder latitude box plot')\n",
    "sns.kdeplot(data=df1_n['lat'])\n",
    "\n",
    "#  subplot #2\n",
    "plt.subplot(122)\n",
    "plt.title('credit card holder latitude violin plot')\n",
    "sns.violinplot(data=df1_n['lat'])\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9f3923c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = plt.figure(figsize=(10,4))\n",
    "\n",
    "#  subplot #1\n",
    "plt.subplot(121)\n",
    "plt.title('credit card holder longitude box plot')\n",
    "sns.kdeplot(data=df1_n['long'])\n",
    "\n",
    "#  subplot #2\n",
    "plt.subplot(122)\n",
    "plt.title('credit card holder longitude violin plot')\n",
    "sns.violinplot(data=df1_n['long'])\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "585b1df6",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = plt.figure(figsize=(10,4))\n",
    "\n",
    "#  subplot #1\n",
    "plt.subplot(121)\n",
    "plt.title('Transaction Amount boxplot')\n",
    "sns.kdeplot(data=df1_n['amt'])\n",
    "\n",
    "#  subplot #2\n",
    "plt.subplot(122)\n",
    "plt.title('transaction amount violin plot')\n",
    "sns.violinplot(data=df1_n['lat'])\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ffb421a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = plt.figure(figsize=(10,4))\n",
    "\n",
    "#  subplot #1\n",
    "plt.subplot(121)\n",
    "plt.title('merchant latitude box plot')\n",
    "sns.kdeplot(data=df1_n['merch_lat'])\n",
    "\n",
    "#  subplot #2\n",
    "plt.subplot(122)\n",
    "plt.title('merchant latitude violin plot')\n",
    "sns.violinplot(data=df1_n['merch_lat'])\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cbc06521",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = plt.figure(figsize=(10,4))\n",
    "\n",
    "#  subplot #1\n",
    "plt.subplot(121)\n",
    "plt.title('merchant longitude box plot')\n",
    "sns.kdeplot(data=df1_n['merch_long'])\n",
    "\n",
    "#  subplot #2\n",
    "plt.subplot(122)\n",
    "plt.title('merchant longitude violin plot')\n",
    "sns.violinplot(data=df1_n['merch_long'])\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fa7f08db",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = plt.figure(figsize=(10,4))\n",
    "\n",
    "#  subplot #1\n",
    "plt.subplot(121)\n",
    "plt.title('unix time box plot')\n",
    "sns.kdeplot(data=df1_n['unix_time'])\n",
    "\n",
    "#  subplot #2\n",
    "plt.subplot(122)\n",
    "plt.title('unix time violin plot')\n",
    "sns.violinplot(data=df1_n['unix_time'])\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bd3e7e2e",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = plt.figure(figsize=(10,4))\n",
    "\n",
    "#  subplot #1\n",
    "plt.subplot(121)\n",
    "plt.title('city population box plot')\n",
    "sns.kdeplot(data=df1_n['city_pop'])\n",
    "\n",
    "#  subplot #2\n",
    "plt.subplot(122)\n",
    "plt.title('city population violin plot')\n",
    "sns.violinplot(data=df1_n['city_pop'])\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ffb68495",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1_n.columns"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dc04198d",
   "metadata": {},
   "source": [
    "# univariate analysis for categorical columns"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0b8f9d47",
   "metadata": {},
   "source": [
    "# top 10 merchants out of 800"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fac9d229",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1_cat['merchant'].value_counts().head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4e8ff35f",
   "metadata": {},
   "outputs": [],
   "source": [
    "temp = df1[\"merchant\"].value_counts().head(10)\n",
    "df = pd.DataFrame({'Class': temp.index,'values': temp.values})\n",
    "\n",
    "trace = go.Bar(\n",
    "    x = df['Class'],y = df['values'],\n",
    "    name=\"Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)\",\n",
    "    marker=dict(color=\"Red\"),\n",
    "    text=df['values']\n",
    ")\n",
    "data = [trace]\n",
    "layout = dict(title = 'Merchant Category univariate analysis',\n",
    "          xaxis = dict(title = 'merchant', showticklabels=True), \n",
    "          yaxis = dict(title = 'count'),\n",
    "          hovermode = 'closest',width=600\n",
    "         )\n",
    "fig = dict(data=data, layout=layout)\n",
    "iplot(fig, filename='class')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b9d37c94",
   "metadata": {},
   "source": [
    "# category of merchants"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "700960ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1_cat['category'].value_counts().head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7f7286ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "temp = df1[\"category\"].value_counts().head(10)\n",
    "df = pd.DataFrame({'Class': temp.index,'values': temp.values})\n",
    "\n",
    "trace = go.Bar(\n",
    "    x = df['Class'],y = df['values'],\n",
    "    name=\"Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)\",\n",
    "    marker=dict(color=\"Red\"),\n",
    "    text=df['values']\n",
    ")\n",
    "data = [trace]\n",
    "layout = dict(title = 'category univariate analysis',\n",
    "          xaxis = dict(title = 'category', showticklabels=True), \n",
    "          yaxis = dict(title = 'count'),\n",
    "          hovermode = 'closest',width=600\n",
    "         )\n",
    "fig = dict(data=data, layout=layout)\n",
    "iplot(fig, filename='class')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "23b57599",
   "metadata": {},
   "source": [
    "# top 10 cities of credit card holder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a7c2cd4d",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1_cat['city'].value_counts().head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9cfedd78",
   "metadata": {},
   "outputs": [],
   "source": [
    "temp = df1[\"city\"].value_counts().head(10)\n",
    "df = pd.DataFrame({'Class': temp.index,'values': temp.values})\n",
    "\n",
    "trace = go.Bar(\n",
    "    x = df['Class'],y = df['values'],\n",
    "    name=\"Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)\",\n",
    "    marker=dict(color=\"Red\"),\n",
    "    text=df['values']\n",
    ")\n",
    "data = [trace]\n",
    "layout = dict(title = 'city univariate analysis',\n",
    "          xaxis = dict(title = 'city', showticklabels=True), \n",
    "          yaxis = dict(title = 'count'),\n",
    "          hovermode = 'closest',width=600\n",
    "         )\n",
    "fig = dict(data=data, layout=layout)\n",
    "iplot(fig, filename='class')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "12533f7f",
   "metadata": {},
   "source": [
    "# top 10 state of credit card holder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "239bd445",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1_cat['state'].value_counts().head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0a9909d2",
   "metadata": {},
   "outputs": [],
   "source": [
    "temp = df1[\"state\"].value_counts().head(10)\n",
    "df = pd.DataFrame({'Class': temp.index,'values': temp.values})\n",
    "\n",
    "trace = go.Bar(\n",
    "    x = df['Class'],y = df['values'],\n",
    "    name=\"Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)\",\n",
    "    marker=dict(color=\"Red\"),\n",
    "    text=df['values']\n",
    ")\n",
    "data = [trace]\n",
    "layout = dict(title = 'state univariate analysis',\n",
    "          xaxis = dict(title = 'state', showticklabels=True), \n",
    "          yaxis = dict(title = 'count'),\n",
    "          hovermode = 'closest',width=600\n",
    "         )\n",
    "fig = dict(data=data, layout=layout)\n",
    "iplot(fig, filename='class')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4289f607",
   "metadata": {},
   "source": [
    "# top 10 jobs of credit card holder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "93278363",
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "df1_cat['job'].value_counts().head(50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ab1e25ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "temp = df1[\"job\"].value_counts().head(10)\n",
    "df = pd.DataFrame({'Class': temp.index,'values': temp.values})\n",
    "\n",
    "trace = go.Bar(\n",
    "    x = df['Class'],y = df['values'],\n",
    "    name=\"Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)\",\n",
    "    marker=dict(color=\"Red\"),\n",
    "    text=df['values']\n",
    ")\n",
    "data = [trace]\n",
    "layout = dict(title = 'job univariate analysis',\n",
    "          xaxis = dict(title = 'job', showticklabels=True), \n",
    "          yaxis = dict(title = 'count'),\n",
    "          hovermode = 'closest',width=600\n",
    "         )\n",
    "fig = dict(data=data, layout=layout)\n",
    "iplot(fig, filename='class')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e7dd9ef4",
   "metadata": {},
   "source": [
    "# statistical insights"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dad41870",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "802c3255",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1.std()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fc980d6f",
   "metadata": {},
   "source": [
    "# checking the data imbalance ratio"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b04ff904",
   "metadata": {},
   "outputs": [],
   "source": [
    "target0 = df1.loc[df1['is_fraud']==0]\n",
    "target1 = df1.loc[df1['is_fraud']==1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a6cef5b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1['is_fraud'].value_counts()\n",
    "(df1['is_fraud'].value_counts()/len(df1['is_fraud']))*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3c6083c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "round(len(target0)/len(target1),2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bc9a15da",
   "metadata": {},
   "outputs": [],
   "source": [
    "m=df1['is_fraud'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "45f23861",
   "metadata": {},
   "outputs": [],
   "source": [
    "import plotly.express as px\n",
    "import numpy\n",
    " \n",
    "# Random Data\n",
    "random_x = m.values\n",
    "names = m.index\n",
    " \n",
    "fig = px.pie(values=random_x, names=names,title='Credit card transaction -data imbalance')\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fcfcc77b",
   "metadata": {},
   "outputs": [],
   "source": [
    "temp = df1[\"is_fraud\"].value_counts()\n",
    "df = pd.DataFrame({'Class': temp.index,'values': temp.values})\n",
    "\n",
    "trace = go.Bar(\n",
    "    x = df['Class'],y = df['values'],\n",
    "    name=\"Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)\",\n",
    "    marker=dict(color=\"Red\"),\n",
    "    text=df['values']\n",
    ")\n",
    "data = [trace]\n",
    "layout = dict(title = 'Credit Card Fraud Class - data unbalance (Notfraud= 0,Fraud =1)',\n",
    "          xaxis = dict(title = 'Class', showticklabels=True), \n",
    "          yaxis = dict(title = 'Number of transactions'),\n",
    "          hovermode = 'closest',width=600\n",
    "         )\n",
    "fig = dict(data=data, layout=layout)\n",
    "iplot(fig, filename='class')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "03a97020",
   "metadata": {},
   "source": [
    "# inference:\n",
    "    df1 dataframe that is application data is highly imbalanced .\n",
    "    'defaulted population is 0.57% and non defaulted population ois 99.4%'"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6542d240",
   "metadata": {},
   "source": [
    "we will seperately analyse the data based in the target variable for a better understanding"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1ad777da",
   "metadata": {},
   "source": [
    "# bivariate analysis "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9acd429d",
   "metadata": {},
   "source": [
    "# gender percenatge"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "66f96244",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_gender=(df1['gender'].value_counts()/len(df1['gender']))*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b2b5c64f",
   "metadata": {},
   "outputs": [],
   "source": [
    "temp = df1[\"gender\"].value_counts()\n",
    "df = pd.DataFrame({'Class': temp.index,'values': temp.values})\n",
    "\n",
    "trace = go.Bar(\n",
    "    x = df['Class'],y = df['values'],\n",
    "    name=\"Credit Card Transaction - Gender Wise Analysis\",\n",
    "    marker=dict(color=\"Green\"),\n",
    "    text=df['values']\n",
    ")\n",
    "data = [trace]\n",
    "layout = dict(title = \"Credit Card Transaction - Gender Wise Analysis\",\n",
    "          xaxis = dict(title = 'Gender', showticklabels=True), \n",
    "          yaxis = dict(title = 'Number of transactions'),\n",
    "          hovermode = 'closest',width=600\n",
    "         )\n",
    "fig = dict(data=data, layout=layout)\n",
    "iplot(fig, filename='class')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c2148515",
   "metadata": {},
   "outputs": [],
   "source": [
    "pd.crosstab(target0['gender'],target0['is_fraud'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c63342f",
   "metadata": {},
   "outputs": [],
   "source": [
    "target0['gender'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "486a7436",
   "metadata": {},
   "outputs": [],
   "source": [
    "target0.groupby(\"gender\")[\"is_fraud\"].count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "73cd58f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "(target0.groupby(\"gender\")[\"is_fraud\"].count())/(len(df1['is_fraud']))*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f82d38de",
   "metadata": {},
   "outputs": [],
   "source": [
    "(target1.groupby(\"gender\")[\"is_fraud\"].count())/(len(df1['is_fraud']))*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c6fb2da",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.rcParams['figure.figsize']=(15,5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bbd860de",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.subplot(121)\n",
    "sns.countplot(x='is_fraud',hue='gender',data=target0,palette='Set2')\n",
    "plt.title('Gender distribution in not fraud transactions')\n",
    "plt.ylabel('Not fraud transactions')\n",
    "plt.subplot(122)\n",
    "sns.countplot(x='is_fraud',hue='gender',data=target1,palette='rocket')\n",
    "plt.title('Gender distribution in fraud transactions')\n",
    "plt.ylabel('Fraud transactions')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6c9f9caa",
   "metadata": {},
   "source": [
    "# insights "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "86aa8408",
   "metadata": {},
   "source": [
    "*it seems like female clients done more transaction than male customers\n",
    "*54.45% female clients are non defaulters while 44.96% male clients are non defaulters\n",
    "*0.28% female clients are defaulters and 0.29% male clients are defaulters"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7e430137",
   "metadata": {},
   "source": [
    "# binning dob "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3234a0e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import datetime"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8dfcd488",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1['dob']=pd.to_datetime(df1['dob'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b866c957",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a6dbc5f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['dob']=pd.to_datetime(df['dob'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2c920d76",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1['dob']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6a642520",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1['year']=pd.DatetimeIndex(df1['dob']).year"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "69f572bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['year']=pd.DatetimeIndex(df['dob']).year"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bab38ae1",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1['year']=2022-df1['year']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "85d5a878",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['year']=2022-df['year']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f3ea8a50",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "205a8afd",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1['year'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ac53ca1b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def function_mrp(a):\n",
    "    if a <=25 :\n",
    "        return 'Very young age'\n",
    "    if a > 25 and a <= 35:\n",
    "        return 'Young age'\n",
    "    if a > 35 and a <= 45:\n",
    "        return 'Middle age'\n",
    "    else:\n",
    "        return 'Senior citizen'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c020faef",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1['age_group'] = df1['year'].apply(function_mrp)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "198fbcac",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['age_group'] = df['year'].apply(function_mrp)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "07babb7b",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "43204ffc",
   "metadata": {},
   "outputs": [],
   "source": [
    "target0 = df1.loc[df1['is_fraud']==0]\n",
    "target1 = df1.loc[df1['is_fraud']==1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "43a2c979",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "87d18feb",
   "metadata": {},
   "outputs": [],
   "source": [
    "target0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f9206327",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(15,6))\n",
    "plt.subplot(121)\n",
    "sns.countplot(x='is_fraud',hue='age_group',data=target0,palette='Set2')\n",
    "plt.title('Age group vs not fraud transactions')\n",
    "plt.ylabel('Not fraud transactions')\n",
    "plt.subplot(122)\n",
    "sns.countplot(x='is_fraud',hue='age_group',data=target1,palette='Set2')\n",
    "plt.title('Age group vs Fraud transactions')\n",
    "plt.ylabel('Fraud transactions')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e28abd33",
   "metadata": {},
   "source": [
    "Insights\n",
    "*senior citizen above 60 are higher than any other in case of defaulters as well as non defaulters\n",
    "*also senior citizens age group facing paying difficulties are the most\n",
    "*while middle age group and very young age group facing less difficulties in paying"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9f79bc30",
   "metadata": {},
   "source": [
    "# job distribution based on target0 and target1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b135817e",
   "metadata": {},
   "outputs": [],
   "source": [
    "temp = target0[\"job\"].value_counts().head(10)\n",
    "df = pd.DataFrame({'Class': temp.index,'values': temp.values})\n",
    "\n",
    "trace = go.Bar(\n",
    "    x = df['Class'],y = df['values'],\n",
    "    name=\"Credit Card Transaction - job Wise Analysis target0\",\n",
    "    marker=dict(color=\"blue\"),\n",
    "    text=df['values']\n",
    ")\n",
    "data = [trace]\n",
    "layout = dict(title =\"Credit Card Transaction - job Wise Analysis target0\",\n",
    "          xaxis = dict(title = 'job', showticklabels=True), \n",
    "          yaxis = dict(title = 'Number of transactions'),\n",
    "          hovermode = 'closest',width=600\n",
    "         )\n",
    "fig = dict(data=data, layout=layout)\n",
    "iplot(fig, filename='class')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b1465d3b",
   "metadata": {},
   "source": [
    "#inference\n",
    "1.jobs such as film editor , agriculture consultant, financial trader are quitely the non defaulters in credit card transaction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bbcd31d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "temp = target1[\"job\"].value_counts().head(10)\n",
    "df = pd.DataFrame({'Class': temp.index,'values': temp.values})\n",
    "\n",
    "trace = go.Bar(\n",
    "    x = df['Class'],y = df['values'],\n",
    "    name=\"Credit Card Transaction - job Wise Analysis target1\",\n",
    "    marker=dict(color=\"red\"),\n",
    "    text=df['values']\n",
    ")\n",
    "data = [trace]\n",
    "layout = dict(title =\"job category wise analysis  for fraud transaction\",\n",
    "          xaxis = dict(title = 'job categories', showticklabels=True), \n",
    "          yaxis = dict(title = 'Number of transactions'),\n",
    "          hovermode = 'closest',width=600\n",
    "         )\n",
    "fig = dict(data=data, layout=layout)\n",
    "iplot(fig, filename='class')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "51ae0106",
   "metadata": {},
   "source": [
    "# inference\n",
    "1.jobs such as materials engineer , podiatrist, energy engineer are quitely the defaulters in credit card transaction"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bfb35c9e",
   "metadata": {},
   "source": [
    "# merchant category vs fraud or not fraud"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ff35c0ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_category=pd.crosstab(df1['category'],df1['is_fraud'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "608fc751",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_c=df_category.reset_index()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "61032258",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_c.rename(columns={'is_fraud':'index',0:'not_fraud',1:'fraud'},inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c02c77c1",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_c=df_c.sort_values('fraud',ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "15e174e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "from plotly.subplots import make_subplots \n",
    "import plotly.graph_objects as go\n",
    "\n",
    "\n",
    "fig = make_subplots(rows=1, cols=2, shared_yaxes=True,subplot_titles=('Merchant category vs not fraud counts',\"Merchant  category vs fraud counts\"))\n",
    "\n",
    "fig.add_trace(go.Bar(x=df_c['category'], y=df_c['not_fraud'],\n",
    "                    marker=dict(color=df_c['not_fraud'], coloraxis=\"coloraxis\")),\n",
    "              1, 1)\n",
    "\n",
    "fig.add_trace(go.Bar(x=df_c['category'], y=df_c['fraud'],\n",
    "                    marker=dict( color=df_c['fraud'],coloraxis=\"coloraxis\")),\n",
    "              1, 2)\n",
    "\n",
    "fig['layout']['xaxis']['title']='merchant categories'\n",
    "fig['layout']['xaxis2']['title']='merchant categories'\n",
    "fig['layout']['yaxis']['title']='not fraud counts'\n",
    "fig['layout']['yaxis2']['title']='fraud counts'\n",
    "\n",
    "fig.update_layout(coloraxis=dict(colorscale='viridis'), showlegend=False)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9355924d",
   "metadata": {},
   "source": [
    "# insights \n",
    "*merchant categories such as gas_transport,home,grocery_pos, shopping_pos has high non fraud transactions\n",
    "*merchant categories such as grocery_pos,shopping_net,misc_net, shopping_pos has fraud transactions"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "483da801",
   "metadata": {},
   "source": [
    "# cities of credit card holder vs fraud or not fraud counts "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5c6aa722",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1['city'].value_counts().head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f86d0bbf",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_city=pd.crosstab(df1['city'],df1['is_fraud'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "02800314",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_city.reset_index(inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "53777db3",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_city.rename(columns={0:'not_fraud',1:'fraud'},inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1c947b6e",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_city=df_city.sort_values('fraud',ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "775dfa73",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_city = df_city.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c73b2817",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_city"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f2b2eb31",
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "from plotly.subplots import make_subplots \n",
    "import plotly.graph_objects as go\n",
    "\n",
    "\n",
    "fig = make_subplots(rows=1, cols=2, shared_yaxes=True,subplot_titles=('cities of credit card holder vs not fraud counts',\"cities of credit card holder vs fraud counts\"))\n",
    "\n",
    "fig.add_trace(go.Bar(x=df_city['city'], y=df_city['not_fraud'],\n",
    "                    marker=dict(color=df_city['not_fraud'], coloraxis=\"coloraxis\")),\n",
    "              1, 1)\n",
    "\n",
    "fig.add_trace(go.Bar(x=df_city['city'], y=df_city['fraud'],\n",
    "                    marker=dict( color=df_city['fraud'],coloraxis=\"coloraxis\")),\n",
    "              1, 2)\n",
    "\n",
    "fig['layout']['xaxis']['title']='cities of credit card holder'\n",
    "fig['layout']['xaxis2']['title']='cities of credit card holder'\n",
    "fig['layout']['yaxis']['title']='not fraud counts'\n",
    "fig['layout']['yaxis2']['title']='fraud counts'\n",
    "\n",
    "fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0fac6d5b",
   "metadata": {},
   "source": [
    "# insights:\n",
    "1.we can clearly see that san Antonio city has the highest non fraud transaction where as huston city has the highest fraud transaction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7805dacd",
   "metadata": {},
   "outputs": [],
   "source": [
    "temp = target0[\"city\"].value_counts().head(10)\n",
    "df = pd.DataFrame({'Class': temp.index,'values': temp.values})\n",
    "\n",
    "trace = go.Bar(\n",
    "    x = df['Class'],y = df['values'],\n",
    "    name='cities of credit card holder vs not fraud counts',\n",
    "    marker=dict(color=\"green\"),\n",
    "    text=df['values']\n",
    ")\n",
    "data = [trace]\n",
    "layout = dict(title ='cities of credit card holder vs not fraud counts',\n",
    "          xaxis = dict(title = 'city', showticklabels=True), \n",
    "          yaxis = dict(title = 'Number of transactions'),\n",
    "          hovermode = 'closest',width=600\n",
    "         )\n",
    "fig = dict(data=data, layout=layout)\n",
    "iplot(fig, filename='class')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5664d885",
   "metadata": {},
   "outputs": [],
   "source": [
    "temp = target1[\"city\"].value_counts().head(10)\n",
    "df = pd.DataFrame({'Class': temp.index,'values': temp.values})\n",
    "\n",
    "trace = go.Bar(\n",
    "    x = df['Class'],y = df['values'],\n",
    "    name='cities of credit card holder vs  fraud counts',\n",
    "    marker=dict(color=\"red\"),\n",
    "    text=df['values']\n",
    ")\n",
    "data = [trace]\n",
    "layout = dict(title ='cities of credit card holder vs fraud counts',\n",
    "          xaxis = dict(title = 'city', showticklabels=True), \n",
    "          yaxis = dict(title = 'Number of transactions'),\n",
    "          hovermode = 'closest',width=600\n",
    "         )\n",
    "fig = dict(data=data, layout=layout)\n",
    "iplot(fig, filename='class')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "84fbeee4",
   "metadata": {},
   "source": [
    "# top 10 merchants vs fraud transactions and not fraud transactions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "40f68199",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.rcParams['figure.figsize']=(15,8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4c3b8b7d",
   "metadata": {},
   "outputs": [],
   "source": [
    "temp = target0[\"merchant\"].value_counts().head(10)\n",
    "df = pd.DataFrame({'Class': temp.index,'values': temp.values})\n",
    "\n",
    "trace = go.Bar(\n",
    "    x = df['Class'],y = df['values'],\n",
    "    name=' top 10 merchants vs not fraud transactions',\n",
    "    marker=dict(color=\"green\"),\n",
    "    text=df['values']\n",
    ")\n",
    "data = [trace]\n",
    "layout = dict(title ='top 10 merchants vs not fraud transactions',\n",
    "          xaxis = dict(title = 'merchant', showticklabels=True), \n",
    "          yaxis = dict(title = 'Number of non fraud transactions'),\n",
    "          hovermode = 'closest',width=600\n",
    "         )\n",
    "fig = dict(data=data, layout=layout)\n",
    "iplot(fig, filename='class')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "501c0211",
   "metadata": {},
   "outputs": [],
   "source": [
    "temp = target1[\"merchant\"].value_counts().head(10)\n",
    "df = pd.DataFrame({'Class': temp.index,'values': temp.values})\n",
    "\n",
    "trace = go.Bar(\n",
    "    x = df['Class'],y = df['values'],\n",
    "    name=' top 10 merchants vs  fraud transactions',\n",
    "    marker=dict(color=\"Red\"),\n",
    "    text=df['values']\n",
    ")\n",
    "data = [trace]\n",
    "layout = dict(title ='top 10 merchants vs  fraud transactions',\n",
    "          xaxis = dict(title = 'merchant', showticklabels=True), \n",
    "          yaxis = dict(title = 'Number of fraud transactions'),\n",
    "          hovermode = 'closest',width=600\n",
    "         )\n",
    "fig = dict(data=data, layout=layout)\n",
    "iplot(fig, filename='class')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7489d768",
   "metadata": {},
   "source": [
    "# insights "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4946a6ac",
   "metadata": {},
   "source": [
    "*merchants like kilback llc, schumm plc , cormier llc have high non fraud transcations \n",
    "*mercahnts like rau and sons , kozey boehm , cormier llc have high fraud transactions "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "95205069",
   "metadata": {},
   "source": [
    "# top 10 states vs fraud transactions and not fraud transactions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7b077906",
   "metadata": {},
   "outputs": [],
   "source": [
    "temp = target0[\"state\"].value_counts().head(10)\n",
    "df = pd.DataFrame({'Class': temp.index,'values': temp.values})\n",
    "\n",
    "trace = go.Bar(\n",
    "    x = df['Class'],y = df['values'],\n",
    "    name=' top 10 merchants vs not fraud transactions',\n",
    "    marker=dict(color=\"green\"),\n",
    "    text=df['values']\n",
    ")\n",
    "data = [trace]\n",
    "layout = dict(title ='top 10 states vs not fraud transactions',\n",
    "          xaxis = dict(title = 'merchant', showticklabels=True), \n",
    "          yaxis = dict(title = 'Number of non fraud transactions'),\n",
    "          hovermode = 'closest',width=600\n",
    "         )\n",
    "fig = dict(data=data, layout=layout)\n",
    "iplot(fig, filename='class')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "881d6866",
   "metadata": {},
   "outputs": [],
   "source": [
    "temp = target1[\"state\"].value_counts().head(10)\n",
    "df = pd.DataFrame({'Class': temp.index,'values': temp.values})\n",
    "\n",
    "trace = go.Bar(\n",
    "    x = df['Class'],y = df['values'],\n",
    "    name=' top 10 merchants vs  fraud transactions',\n",
    "    marker=dict(color=\"Red\"),\n",
    "    text=df['values']\n",
    ")\n",
    "data = [trace]\n",
    "layout = dict(title ='Top 10 states vs  Fraud transactions',\n",
    "          xaxis = dict(title = 'States', showticklabels=True), \n",
    "          yaxis = dict(title = 'Number of fraud transactions'),\n",
    "          hovermode = 'closest',width=600\n",
    "         )\n",
    "fig = dict(data=data, layout=layout)\n",
    "iplot(fig, filename='class')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "695fd9fc",
   "metadata": {},
   "source": [
    "# insights "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "54a74cfb",
   "metadata": {},
   "source": [
    "*state tx,ny,pa has high fraud transactions as well as high non fraud transactions "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8616860c",
   "metadata": {},
   "source": [
    "# amt vs non fraud transaction distribution and amt vs fraud transaction distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8f20d925",
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.set(style=\"darkgrid\")\n",
    "plt.figure(figsize=(40,20))\n",
    "    \n",
    "plt.subplot(1,2,1)                                   \n",
    "sns.distplot(target0['amt'], color=\"g\" )\n",
    "plt.yscale('linear') \n",
    "plt.xlabel('amt', fontsize= 30, fontweight=\"bold\")\n",
    "plt.ylabel('Non fraud transactions', fontsize= 30, fontweight=\"bold\")                    #Target 0\n",
    "plt.xticks(rotation=90, fontsize=30)\n",
    "plt.yticks(rotation=360, fontsize=30)\n",
    "     \n",
    "    \n",
    "    \n",
    "    \n",
    "plt.subplot(1,2,2)                                                                                                      \n",
    "sns.distplot(target1['amt'], color=\"r\")\n",
    "plt.yscale('linear')    \n",
    "plt.xlabel('amt', fontsize= 30, fontweight=\"bold\")\n",
    "plt.ylabel('Fraud Transcations', fontsize= 30, fontweight=\"bold\")                       # Target 1\n",
    "plt.xticks(rotation=90, fontsize=30)\n",
    "plt.yticks(rotation=360, fontsize=30)\n",
    "    \n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "08c67f12",
   "metadata": {},
   "outputs": [],
   "source": [
    "target0['amt'].std()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "da18a902",
   "metadata": {},
   "outputs": [],
   "source": [
    "target1['amt'].std()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "807c4ac4",
   "metadata": {},
   "outputs": [],
   "source": [
    "target0['amt'].skew() #highle positive skewed  "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "47bea3b4",
   "metadata": {},
   "source": [
    "# insight"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "84eba6c6",
   "metadata": {},
   "source": [
    "Dist. plot highlights the curve shape which is wider for Target 1 in comparison to Target 0 which is narrower with well-defined edges."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "870f008b",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1['trans_date_trans_time']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "203ffddb",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1['trans_date_trans_time']=pd.to_datetime(df1['trans_date_trans_time'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6d3448b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['trans_date_trans_time']=pd.to_datetime(df['trans_date_trans_time'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cb3623ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1['weekday_no'] = df1['trans_date_trans_time'].dt.dayofweek\n",
    "df1['week_day'] = df1['trans_date_trans_time'].dt.day_name()\n",
    "df1['week_no'] = df1['trans_date_trans_time'].dt.week\n",
    "df1['day_no'] = df1['trans_date_trans_time'].dt.day\n",
    "df1['min_day'] = df1['trans_date_trans_time'].dt.minute\n",
    "df1['hr_day'] = df1['trans_date_trans_time'].dt.hour\n",
    "df1['month_name'] = df1['trans_date_trans_time'].dt.month_name()\n",
    "df1['month'] = df1['trans_date_trans_time'].dt.month\n",
    "df1['year'] = df1['trans_date_trans_time'].dt.year\n",
    "df1['year_dayno'] = df1['trans_date_trans_time'].dt.dayofyear"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f53a31d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['weekday_no'] = df['trans_date_trans_time'].dt.dayofweek\n",
    "df['week_day'] = df['trans_date_trans_time'].dt.day_name()\n",
    "df['week_no'] = df['trans_date_trans_time'].dt.week\n",
    "df['day_no'] = df['trans_date_trans_time'].dt.day\n",
    "df['min_day'] = df['trans_date_trans_time'].dt.minute\n",
    "df['hr_day'] = df['trans_date_trans_time'].dt.hour\n",
    "df['month_name'] = df['trans_date_trans_time'].dt.month_name()\n",
    "df['month'] = df['trans_date_trans_time'].dt.month\n",
    "df['year'] = df['trans_date_trans_time'].dt.year\n",
    "df['year_dayno'] = df['trans_date_trans_time'].dt.dayofyear"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5e41bd8",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "71c64701",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "537a1dfd",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1['week_day'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "956f602b",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1['month_name'].value_counts().sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "796bdc1c",
   "metadata": {},
   "outputs": [],
   "source": [
    "temp = df1[\"week_day\"].value_counts().head(10)\n",
    "df = pd.DataFrame({'Class': temp.index,'values': temp.values})\n",
    "\n",
    "trace = go.Bar(\n",
    "    x = df['Class'],y = df['values'],\n",
    "    name=' week days vs  no of transactions',\n",
    "    marker=dict(color=\"Red\"),\n",
    "    text=df['values']\n",
    ")\n",
    "data = [trace]\n",
    "layout = dict(title ='week days vs no of transactions',\n",
    "          xaxis = dict(title = 'weekdays', showticklabels=True), \n",
    "          yaxis = dict(title = 'Number of fraud transactions'),\n",
    "          hovermode = 'closest',width=600\n",
    "         )\n",
    "fig = dict(data=data, layout=layout)\n",
    "iplot(fig, filename='class')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ef899557",
   "metadata": {},
   "outputs": [],
   "source": [
    "temp = df1['month_name'].value_counts().sort_values(ascending=False)\n",
    "df = pd.DataFrame({'Class': temp.index,'values': temp.values})\n",
    "\n",
    "trace = go.Bar(\n",
    "    x = df['Class'],y = df['values'],\n",
    "    name=' month vs  no of transactions',\n",
    "    marker=dict(color=\"Red\"),\n",
    "    text=df['values']\n",
    ")\n",
    "data = [trace]\n",
    "layout = dict(title ='month vs no of transactions',\n",
    "          xaxis = dict(title = '', showticklabels=True), \n",
    "          yaxis = dict(title = 'Number of fraud transactions'),\n",
    "          hovermode = 'closest',width=600\n",
    "         )\n",
    "fig = dict(data=data, layout=layout)\n",
    "iplot(fig, filename='class')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "59200ba9",
   "metadata": {},
   "source": [
    "# Bivariate Analysis : Numerical and Categorical wrt target variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "24b3f6ad",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4bddf3c2",
   "metadata": {},
   "source": [
    "# merchant vs amount transaction vs target variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f64a29c0",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_merch_amt=df1.pivot_table(index='merchant',columns='is_fraud',values='amt',aggfunc='sum')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ebb60d8f",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_merch_amt.reset_index(inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a15759b6",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_merch_amt.rename(columns={0:'not_fraud',1:'fraud'},inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "39b6d553",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_merch_amt=df_merch_amt.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "03130495",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_merch_amt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6179a8fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "from plotly.subplots import make_subplots \n",
    "import plotly.graph_objects as go\n",
    "\n",
    "\n",
    "fig = make_subplots(rows=1, cols=2, shared_yaxes=True,subplot_titles=('merchant non-fraud transaction amount',\"merchant fraud transaction amount\"))\n",
    "\n",
    "fig.add_trace(go.Bar(x=df_merch_amt['merchant'], y=df_merch_amt['not_fraud'],\n",
    "                    marker=dict(color=df_merch_amt['not_fraud'], coloraxis=\"coloraxis\")),\n",
    "              1, 1)\n",
    "\n",
    "fig.add_trace(go.Bar(x=df_merch_amt['merchant'], y=df_merch_amt['fraud'],\n",
    "                    marker=dict( color=df_merch_amt['fraud'],coloraxis=\"coloraxis\")),\n",
    "              1, 2)\n",
    "\n",
    "fig['layout']['xaxis']['title']='merchant'\n",
    "fig['layout']['xaxis2']['title']='merchant'\n",
    "fig['layout']['yaxis']['title']='not fraud transaction amount'\n",
    "fig['layout']['yaxis2']['title']='fraud transaction amount'\n",
    "\n",
    "fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4409d780",
   "metadata": {},
   "source": [
    "# job vs amount transaction vs target variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "96065213",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_job_amt=df1.pivot_table(index='job',columns='is_fraud',values='amt',aggfunc='sum')\n",
    "df_job_amt.reset_index(inplace=True)\n",
    "df_job_amt.rename(columns={0:'not_fraud',1:'fraud'},inplace=True)\n",
    "df_job_amt=df_job_amt.sort_values('fraud',ascending=False).head(10)\n",
    "df_job_amt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "55158196",
   "metadata": {},
   "outputs": [],
   "source": [
    "from plotly.subplots import make_subplots \n",
    "import plotly.graph_objects as go\n",
    "\n",
    "\n",
    "fig = make_subplots(rows=1, cols=2, shared_yaxes=True,subplot_titles=('job non-fraud transaction amount',\"job fraud transaction amount\"))\n",
    "\n",
    "fig.add_trace(go.Bar(x=df_job_amt['job'], y=df_job_amt['not_fraud'],\n",
    "                    marker=dict(color=df_job_amt['not_fraud'], coloraxis=\"coloraxis\")),\n",
    "              1, 1)\n",
    "\n",
    "fig.add_trace(go.Bar(x=df_job_amt['job'], y=df_job_amt['fraud'],\n",
    "                    marker=dict( color=df_job_amt['fraud'],coloraxis=\"coloraxis\")),\n",
    "              1, 2)\n",
    "\n",
    "fig['layout']['xaxis']['title']='jobs'\n",
    "fig['layout']['xaxis2']['title']='jobs'\n",
    "fig['layout']['yaxis']['title']='not fraud transaction amount'\n",
    "fig['layout']['yaxis2']['title']='fraud transaction amount'\n",
    "\n",
    "fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "545064a9",
   "metadata": {},
   "source": [
    "# month vs amount transaction vs target variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d12fccc5",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1['month_name'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2b2e1b54",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_month_amt=df1.pivot_table(index='month_name',columns='is_fraud',values='amt',aggfunc='sum')\n",
    "df_month_amt.reset_index(inplace=True)\n",
    "df_month_amt.rename(columns={0:'not_fraud',1:'fraud'},inplace=True)\n",
    "df_month_amt=df_month_amt.head(12)\n",
    "df_month_amt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "812c9940",
   "metadata": {},
   "outputs": [],
   "source": [
    "from plotly.subplots import make_subplots \n",
    "import plotly.graph_objects as go\n",
    "\n",
    "\n",
    "fig = make_subplots(rows=1, cols=2, shared_yaxes=True,subplot_titles=('month wise non-fraud transaction amount',\"month wise fraud transaction amount\"))\n",
    "\n",
    "fig.add_trace(go.Bar(x=df_month_amt['month_name'], y=df_month_amt['not_fraud'],\n",
    "                    marker=dict(color=df_month_amt['not_fraud'], coloraxis=\"coloraxis\")),\n",
    "              1, 1)\n",
    "\n",
    "fig.add_trace(go.Bar(x=df_month_amt['month_name'], y=df_month_amt['fraud'],\n",
    "                    marker=dict( color=df_month_amt['fraud'],coloraxis=\"coloraxis\")),\n",
    "              1, 2)\n",
    "\n",
    "fig['layout']['xaxis']['title']='months'\n",
    "fig['layout']['xaxis2']['title']='months'\n",
    "fig['layout']['yaxis']['title']='not fraud transaction amount'\n",
    "fig['layout']['yaxis2']['title']='fraud transaction amount'\n",
    "\n",
    "fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ba27acbc",
   "metadata": {},
   "source": [
    "# Merchant Category vs amt transaction vs target variable "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8a7f3e08",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_mercat_amt=df1.pivot_table(index='category',columns='is_fraud',values='amt',aggfunc='sum')\n",
    "df_mercat_amt.reset_index(inplace=True)\n",
    "df_mercat_amt.rename(columns={0:'not_fraud',1:'fraud'},inplace=True)\n",
    "df_mercat_amt=df_mercat_amt.sort_values('fraud',ascending=False).head(10)\n",
    "df_mercat_amt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9df427f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "from plotly.subplots import make_subplots \n",
    "import plotly.graph_objects as go\n",
    "\n",
    "\n",
    "fig = make_subplots(rows=1, cols=2, shared_yaxes=True,subplot_titles=('merchant category wise non-fraud transaction amount',\"merchant category wise fraud transaction amount\"))\n",
    "\n",
    "fig.add_trace(go.Bar(x=df_mercat_amt['category'], y=df_mercat_amt['not_fraud'],\n",
    "                    marker=dict(color=df_mercat_amt['not_fraud'], coloraxis=\"coloraxis\")),\n",
    "              1, 1)\n",
    "\n",
    "fig.add_trace(go.Bar(x=df_mercat_amt['category'], y=df_mercat_amt['fraud'],\n",
    "                    marker=dict( color=df_mercat_amt['fraud'],coloraxis=\"coloraxis\")),\n",
    "              1, 2)\n",
    "\n",
    "fig['layout']['xaxis']['title']='merchant category'\n",
    "fig['layout']['xaxis2']['title']='merchant category'\n",
    "fig['layout']['yaxis']['title']='not fraud transaction amount'\n",
    "fig['layout']['yaxis2']['title']='fraud transaction amount'\n",
    "\n",
    "fig.update_layout(coloraxis=dict(colorscale='viridis'), showlegend=False)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bb5d6be3",
   "metadata": {},
   "source": [
    "# state vs amt transaction vs target variable "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "da997923",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_state_amt=df1.pivot_table(index='state',columns='is_fraud',values='amt',aggfunc='sum')\n",
    "df_state_amt.reset_index(inplace=True)\n",
    "df_state_amt.rename(columns={0:'not_fraud',1:'fraud'},inplace=True)\n",
    "df_state_amt=df_state_amt.sort_values('fraud',ascending=False).head(10)\n",
    "df_state_amt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "054fffeb",
   "metadata": {},
   "outputs": [],
   "source": [
    "from plotly.subplots import make_subplots \n",
    "import plotly.graph_objects as go\n",
    "\n",
    "\n",
    "fig = make_subplots(rows=1, cols=2, shared_yaxes=True,subplot_titles=('State wise non-fraud transaction amount',\"State wise fraud transaction amount\"))\n",
    "\n",
    "fig.add_trace(go.Bar(x=df_state_amt['state'], y=df_state_amt['not_fraud'],\n",
    "                    marker=dict(color=df_state_amt['not_fraud'], coloraxis=\"coloraxis\")),\n",
    "              1, 1)\n",
    "\n",
    "fig.add_trace(go.Bar(x=df_state_amt['state'], y=df_state_amt['fraud'],\n",
    "                    marker=dict( color=df_state_amt['fraud'],coloraxis=\"coloraxis\")),\n",
    "              1, 2)\n",
    "\n",
    "fig['layout']['xaxis']['title']='State'\n",
    "fig['layout']['xaxis2']['title']='State'\n",
    "fig['layout']['yaxis']['title']='not fraud transaction amount'\n",
    "fig['layout']['yaxis2']['title']='fraud transaction amount'\n",
    "\n",
    "fig.update_layout(coloraxis=dict(colorscale='thermal'), showlegend=False)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "acacc21c",
   "metadata": {},
   "source": [
    "# city vs amt transaction vs target variable "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1beef946",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "df_city_amt=df1.pivot_table(index='city',columns='is_fraud',values='amt',aggfunc='sum')\n",
    "df_city_amt.reset_index(inplace=True)\n",
    "df_city_amt.rename(columns={0:'not_fraud',1:'fraud'},inplace=True)\n",
    "df_city_amt=df_city_amt.sort_values('fraud',ascending=False).head(10)\n",
    "df_city_amt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "14b095e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "from plotly.subplots import make_subplots \n",
    "import plotly.graph_objects as go\n",
    "\n",
    "\n",
    "fig = make_subplots(rows=1, cols=2, shared_yaxes=True,subplot_titles=('city wise non-fraud transaction amount',\"city wise fraud transaction amount\"))\n",
    "\n",
    "fig.add_trace(go.Bar(x=df_city_amt['city'], y=df_city_amt['not_fraud'],\n",
    "                    marker=dict(color=df_city_amt['not_fraud'], coloraxis=\"coloraxis\")),\n",
    "              1, 1)\n",
    "\n",
    "fig.add_trace(go.Bar(x=df_city_amt['city'], y=df_city_amt['fraud'],\n",
    "                    marker=dict( color=df_city_amt['fraud'],coloraxis=\"coloraxis\")),\n",
    "              1, 2)\n",
    "\n",
    "fig['layout']['xaxis']['title']='city'\n",
    "fig['layout']['xaxis2']['title']='city'\n",
    "fig['layout']['yaxis']['title']='not fraud transaction amount'\n",
    "fig['layout']['yaxis2']['title']='fraud transaction amount'\n",
    "\n",
    "fig.update_layout(coloraxis=dict(colorscale='plasma'), showlegend=False)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dd4fb6ff",
   "metadata": {},
   "source": [
    "# Gender vs amt transaction amount vs target variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2218b030",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_gender_amt=df1.pivot_table(index='gender',columns='is_fraud',values='amt',aggfunc='sum')\n",
    "df_gender_amt.reset_index(inplace=True)\n",
    "df_gender_amt.rename(columns={0:'not_fraud',1:'fraud'},inplace=True)\n",
    "df_gender_amt=df_gender_amt.sort_values('fraud',ascending=False).head(10)\n",
    "df_gender_amt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b3e12c0f",
   "metadata": {},
   "outputs": [],
   "source": [
    "from plotly.subplots import make_subplots \n",
    "import plotly.graph_objects as go\n",
    "\n",
    "\n",
    "fig = make_subplots(rows=1, cols=2, shared_yaxes=True,subplot_titles=('gender wise non-fraud transaction amount',\"gender wise fraud transaction amount\"))\n",
    "\n",
    "fig.add_trace(go.Bar(x=df_gender_amt['gender'], y=df_gender_amt['not_fraud'],\n",
    "                    marker=dict(color=df_gender_amt['not_fraud'], coloraxis=\"coloraxis\")),\n",
    "              1, 1)\n",
    "\n",
    "fig.add_trace(go.Bar(x=df_gender_amt['gender'], y=df_gender_amt['fraud'],\n",
    "                    marker=dict( color=df_gender_amt['fraud'],coloraxis=\"coloraxis\")),\n",
    "              1, 2)\n",
    "\n",
    "fig['layout']['xaxis']['title']='gender'\n",
    "fig['layout']['xaxis2']['title']='gender'\n",
    "fig['layout']['yaxis']['title']='not fraud transaction amount'\n",
    "fig['layout']['yaxis2']['title']='fraud transaction amount'\n",
    "\n",
    "fig.update_layout(coloraxis=dict(colorscale='jet'), showlegend=False)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "20d57e32",
   "metadata": {},
   "source": [
    "# month vs amount vs target variable trend graph"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ddfd73a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3b8320b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1.pivot_table(index=['year','month_name'],columns=['is_fraud'],values='amt',aggfunc='sum').plot(kind='line')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "05bddca1",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_trend=df1.pivot_table(index=['year','month_name','month'],columns=['is_fraud'],values='amt',aggfunc='sum')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "29d41606",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_trend.reset_index(inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "95a45217",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_trend.rename(columns={0:'not_fraud transaction amount',1:'fraud transaction amount'},inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1b3c9579",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_trend"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d87de900",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_trend=df_trend.sort_values('month')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bbf8bf7b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import plotly.express as px\n",
    "\n",
    "df = px.data.gapminder().query(\"continent=='Oceania'\")\n",
    "fig = px.line(df_trend, x=\"month_name\", y='fraud transaction amount', color='year')\n",
    "\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "84f315c0",
   "metadata": {},
   "source": [
    "# insight \n",
    " from the dataset ,\n",
    " 2019 fraud transaction amount got its peak during december month and least at june month\n",
    " 2020 fraud transaction amount got its peak during may month and least during the april month  \n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c221d753",
   "metadata": {},
   "source": [
    "# statistical Significance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "02a857ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "860f370b",
   "metadata": {},
   "source": [
    "# dropping few columns which will not help in analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d2edc4a5",
   "metadata": {},
   "source": [
    "# for categorical columns with sub category more than 2 and categorical variable equal to 2 we use chi square test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8a4bee39",
   "metadata": {},
   "outputs": [],
   "source": [
    "df2=df1.drop(['trans_date_trans_time','cc_num','first','last','street','zip','dob','trans_num','unix_time','week_day','month_name'],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1af50acd",
   "metadata": {},
   "outputs": [],
   "source": [
    "df2.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a2f55a22",
   "metadata": {},
   "outputs": [],
   "source": [
    "df2.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "35855e95",
   "metadata": {},
   "source": [
    "#our target variable is is_fraud categorical variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "45eccf97",
   "metadata": {},
   "outputs": [],
   "source": [
    "categorical columns\n",
    "1.merchant\n",
    "2.category\n",
    "3.gender\n",
    "4.city\n",
    "5.state\n",
    "6.job\n",
    "7.year\n",
    "8.age_group\n",
    "9.weekday_no\n",
    "10.week_no\n",
    "11.day no\n",
    "12.month"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "49c8e281",
   "metadata": {},
   "outputs": [],
   "source": [
    "from scipy import stats"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c52c2379",
   "metadata": {},
   "source": [
    "# merchant vs is_fraud"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8959d7c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "#null: merchant and is_fraud are independent\n",
    "#alter:merchant and is_fraud are dependent"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6b3c6ae6",
   "metadata": {},
   "outputs": [],
   "source": [
    "stats.chi2_contingency(pd.crosstab(df2['merchant'],df2['is_fraud']))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "babb4141",
   "metadata": {},
   "source": [
    "#pvalue is less than significance level we reject null hypothesis\n",
    "concluding that merchant and is_fraud are dependent"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e6dbdf74",
   "metadata": {},
   "source": [
    "# category vs is_fraud"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "05fd5e4c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#null: category and is_fraud are independent\n",
    "#alter:category and is_fraud are dependent"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e59b4d99",
   "metadata": {},
   "outputs": [],
   "source": [
    "stats.chi2_contingency(pd.crosstab(df2['category'],df2['is_fraud']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3cbb8491",
   "metadata": {},
   "outputs": [],
   "source": [
    "#pvalue is less than significance level we reject null hypothesis\n",
    "concluding that category and is_fraud are dependent"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "32a8cc56",
   "metadata": {},
   "source": [
    "# Gender vs is_fraud"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "76bba9fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "#null: gender and is_fraud are independent\n",
    "#alter:gender and is_fraud are dependent"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f3907979",
   "metadata": {},
   "outputs": [],
   "source": [
    "stats.chi2_contingency(pd.crosstab(df2['gender'],df2['is_fraud']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4206e669",
   "metadata": {},
   "outputs": [],
   "source": [
    "#pvalue is less than significance level we reject null hypothesis\n",
    "concluding that gender and is_fraud are dependent"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "acdb667a",
   "metadata": {},
   "source": [
    "# city vs is_fraud"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b2f9139a",
   "metadata": {},
   "outputs": [],
   "source": [
    "#null: gender and is_fraud are independent\n",
    "#alter:gender and is_fraud are dependent"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "999c6c0c",
   "metadata": {},
   "outputs": [],
   "source": [
    "stats.chi2_contingency(pd.crosstab(df2['city'],df2['is_fraud']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "82f9576d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#pvalue is less than significance level we reject null hypothesis\n",
    "concluding that city and is_fraud are dependent"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ea59c7f1",
   "metadata": {},
   "source": [
    "# state vs is_fraud"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ebffd60b",
   "metadata": {},
   "outputs": [],
   "source": [
    "#null: state and is_fraud are independent\n",
    "#alter:state and is_fraud are dependent"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3db9f375",
   "metadata": {},
   "outputs": [],
   "source": [
    "stats.chi2_contingency(pd.crosstab(df2['state'],df2['is_fraud']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "62a56250",
   "metadata": {},
   "outputs": [],
   "source": [
    "#pvalue is less than significance level we reject null hypothesis\n",
    "concluding that state and is_fraud are dependent"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4dab6de8",
   "metadata": {},
   "source": [
    "# job vs is_fraud"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6682ce1e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#null: job and is_fraud are independent\n",
    "#alter:job and is_fraud are dependent\n",
    "stats.chi2_contingency(pd.crosstab(df2['job'],df2['is_fraud']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0fd1690f",
   "metadata": {},
   "outputs": [],
   "source": [
    "#pvalue is less than significance level we reject null hypothesis\n",
    "concluding that job and is_fraud are dependent"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "388b22c4",
   "metadata": {},
   "source": [
    "# year vs is_fraud"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a390cff3",
   "metadata": {},
   "outputs": [],
   "source": [
    "#null: yEAR and is_fraud are independent\n",
    "#alter:year and is_fraud are dependent\n",
    "stats.chi2_contingency(pd.crosstab(df2['year'],df2['is_fraud']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cd740c95",
   "metadata": {},
   "outputs": [],
   "source": [
    "#pvalue is less than significance level we reject null hypothesis\n",
    "concluding that year and is_fraud are dependent"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "044ca71c",
   "metadata": {},
   "source": [
    "# age_group vs is_fraud"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7267efe9",
   "metadata": {},
   "outputs": [],
   "source": [
    "#null: age_group and is_fraud are independent\n",
    "#alter:age_group and is_fraud are dependent\n",
    "stats.chi2_contingency(pd.crosstab(df2['age_group'],df2['is_fraud']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ecbb5d24",
   "metadata": {},
   "outputs": [],
   "source": [
    "#pvalue is less than significance level we reject null hypothesis\n",
    "concluding that age_group and is_fraud are dependent"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b4fee631",
   "metadata": {},
   "source": [
    "# weekday_no vs is_fraud"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cd462c03",
   "metadata": {},
   "outputs": [],
   "source": [
    "#null:weekday_no and is_fraud are independent\n",
    "#alter:weekday_no and is_fraud are dependent\n",
    "stats.chi2_contingency(pd.crosstab(df2['weekday_no'],df2['is_fraud']))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4abd704a",
   "metadata": {},
   "outputs": [],
   "source": [
    "#pvalue is less than significance level we reject null hypothesis\n",
    "concluding that weekday_no and is_fraud are dependent"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0fe1fb57",
   "metadata": {},
   "source": [
    "# week_no vs is_fraud"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "be796396",
   "metadata": {},
   "outputs": [],
   "source": [
    "#null:week_no and is_fraud are independent\n",
    "#alter:week_no and is_fraud are dependent\n",
    "stats.chi2_contingency(pd.crosstab(df2['week_no'],df2['is_fraud']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "701c93d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "#pvalue is less than significance level we reject null hypothesis\n",
    "concluding that week_no and is_fraud are dependent"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c542186a",
   "metadata": {},
   "source": [
    "# day_no vs is_fraud"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1bfaa908",
   "metadata": {},
   "outputs": [],
   "source": [
    "#null:day_no and is_fraud are independent\n",
    "#alter:day_no and is_fraud are dependent\n",
    "stats.chi2_contingency(pd.crosstab(df2['day_no'],df2['is_fraud']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9bdad384",
   "metadata": {},
   "outputs": [],
   "source": [
    "#pvalue is less than significance level we reject null hypothesis\n",
    "concluding that day_no and is_fraud are dependent"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4ff69d94",
   "metadata": {},
   "source": [
    "# month vs is_fraud"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "984eace2",
   "metadata": {},
   "outputs": [],
   "source": [
    "#null:month  and is_fraud are independent\n",
    "#alter:month  and is_fraud are dependent\n",
    "stats.chi2_contingency(pd.crosstab(df2['month'],df2['is_fraud']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b96a7666",
   "metadata": {},
   "outputs": [],
   "source": [
    "#pvalue is less than significance level we reject null hypothesis\n",
    "concluding that month and is_fraud are dependent"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e3e6c8de",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "840fc548",
   "metadata": {},
   "outputs": [],
   "source": [
    "numerical columns\n",
    "1.amt\n",
    "2.lat\n",
    "3.long\n",
    "4.merch_lat\n",
    "5.merch_long\n",
    "6.city_pop\n",
    "7.year_dayno\n",
    "8.min_day\n",
    "9.hr_day"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eb41a5c8",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d688b115",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "13bd73ae",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f5d668ee",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bfa303aa",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "23c5ecda",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "930d0f58",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4ddf5cb6",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "04ccffb5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "02586e36",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ff34afc3",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4f51a003",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "ed1f26c8",
   "metadata": {},
   "source": [
    "# feature Engineering"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ac48fe5b",
   "metadata": {},
   "source": [
    "# 1.-\tWhether any transformations required"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "470cc1ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "#yoe johnson transformations required for our dataset to reduce our skewness"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ae54984e",
   "metadata": {},
   "source": [
    "# 2.dropping the redundant columns "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ed847c85",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize = (30,30))\n",
    "sns.heatmap(df1.corr(),annot = True, cmap=\"GnBu\",fmt='.2f')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "362a05e9",
   "metadata": {},
   "source": [
    "from correlation matrix we can see columns with \n",
    "high multi collinearity is marked with dark blue we can drop either one column based on vif score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "41e69d07",
   "metadata": {},
   "outputs": [],
   "source": [
    "1.zip\n",
    "2.lat\n",
    "3.long\n",
    "4.unix time\n",
    "5.month \n",
    "6.year_dayno\n",
    "7.week_no"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "01766279",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1_no=df1.select_dtypes(include=[np.number])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9fa4c53c",
   "metadata": {},
   "outputs": [],
   "source": [
    "from statsmodels.stats.outliers_influence import variance_inflation_factor\n",
    "X=df1_no\n",
    "vif = pd.DataFrame()\n",
    "vif[\"VIF_Factor\"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]\n",
    "vif[\"Features\"] = X.columns\n",
    "print(vif)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d7c44b5b",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_mb=df1.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "92733de8",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_mb.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "be58c07d",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_mb.drop(['trans_date_trans_time', 'cc_num', 'merchant', 'street', 'city', 'state','first','last',\n",
    "          'zip', 'lat','long', 'job', 'trans_num', 'unix_time', 'month_name', 'year_dayno','dob','week_day','week_no'],1,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e2e1f62a",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(['trans_date_trans_time', 'cc_num', 'merchant', 'street', 'city', 'state','first','last',\n",
    "          'zip', 'lat','long', 'job', 'trans_num', 'unix_time', 'month_name', 'year_dayno','dob','week_day','week_no'],1,inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "11a221a6",
   "metadata": {},
   "source": [
    "# 3.scaling the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12f43bbe",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_target=df_mb['is_fraud']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7240648b",
   "metadata": {},
   "outputs": [],
   "source": [
    "dft_target=df['is_fraud']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ba48c0ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_mb['weekday_no'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a59c2e39",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_mb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "90cf9901",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_num=df_mb[['amt','city_pop','merch_lat','merch_long','day_no','min_day','hr_day']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "856bf03e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import PowerTransformer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "587c489a",
   "metadata": {},
   "outputs": [],
   "source": [
    "PT = PowerTransformer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "461d1221",
   "metadata": {},
   "outputs": [],
   "source": [
    "PT_Yeo = PowerTransformer(method='yeo-johnson')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d248d1b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_yeo= PT_Yeo.fit_transform(df_num)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "34bcb693",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_yeo1=pd.DataFrame(df_yeo,columns=df_num.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c3421e5e",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_yeo1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d0f9b751",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "400f37ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "dft_num=df[['amt','city_pop','merch_lat','merch_long','day_no','min_day','hr_day']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f8d18470",
   "metadata": {},
   "outputs": [],
   "source": [
    "dft_yeo= PT_Yeo.transform(dft_num)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6d229a29",
   "metadata": {},
   "outputs": [],
   "source": [
    "dft_yeo1=pd.DataFrame(dft_yeo,columns=dft_num.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "091b24e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "dft_yeo1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b94398df",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_yeo1['weekday_no']=df_mb['weekday_no']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9dfc90e1",
   "metadata": {},
   "outputs": [],
   "source": [
    "dft_yeo1['weekday_no']=df_mb['weekday_no']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c416c0e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_yeo1['month']=df_mb['month']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "84b0d5b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "dft_yeo1['month']=df_mb['month']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7f00f272",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_yeo1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2e033b37",
   "metadata": {},
   "outputs": [],
   "source": [
    "dft_yeo1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bed76c94",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_encoded=pd.get_dummies(df_mb[['category','gender','year','age_group']],drop_first=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8f6159a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "dft_encoded=pd.get_dummies(df[['category','gender','year','age_group']],drop_first=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "74ce3594",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_cat=df_encoded.drop('year',axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f5089039",
   "metadata": {},
   "outputs": [],
   "source": [
    "dft_cat=dft_encoded.drop('year',axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4992144b",
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "df_cat"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "492f90e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "dft_cat"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f274f100",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_final=pd.concat([df_yeo1,df_cat],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "59df3a73",
   "metadata": {},
   "outputs": [],
   "source": [
    "dft_final=pd.concat([dft_yeo1,dft_cat],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f349833f",
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "df_final\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3c062a4b",
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "dft_final"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "efc3874f",
   "metadata": {},
   "source": [
    "# Assumptions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d7e8c857",
   "metadata": {},
   "outputs": [],
   "source": [
    "1.Independent rows\n",
    "2.Log(odds)is a linear\n",
    "3.no multi collinearity\n",
    "4.lack of strongly influential outliers"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ab6da3e0",
   "metadata": {},
   "source": [
    "# base model"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "796e1d8a",
   "metadata": {},
   "source": [
    "# train test split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9cfe2971",
   "metadata": {},
   "outputs": [],
   "source": [
    "X=df_final\n",
    "xtrain=sm.add_constant(X)\n",
    "ytrain=df_target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fb2de388",
   "metadata": {},
   "outputs": [],
   "source": [
    "X=dft_final\n",
    "xtest=sm.add_constant(X)\n",
    "ytest=dft_target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7a8dbc72",
   "metadata": {},
   "outputs": [],
   "source": [
    "import statsmodels.api as sm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "842f4f67",
   "metadata": {},
   "outputs": [],
   "source": [
    "logreg=sm.Logit(ytrain,xtrain).fit()\n",
    "print(logreg.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d39fc54d",
   "metadata": {},
   "outputs": [],
   "source": [
    "ytrain_prob=logreg.predict(xtrain)\n",
    "ypred_train=[0 if x < 0.5 else 1 for x in ytrain_prob]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a5016d78",
   "metadata": {},
   "outputs": [],
   "source": [
    "ytest_prob=logreg.predict(xtest)\n",
    "ypred_test=[0 if x < 0.5 else 1 for x in ytest_prob]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "367e498b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import roc_curve"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a83ef4a5",
   "metadata": {},
   "outputs": [],
   "source": [
    "fpr,tpr,thresholds=roc_curve(ytrain,ytrain_prob)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "60fcf995",
   "metadata": {},
   "outputs": [],
   "source": [
    "fpr,tpr,thresholds=roc_curve(ytest,ytest_prob)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9f44573e",
   "metadata": {},
   "outputs": [],
   "source": [
    "roc_auc_score(ytrain, ytrain_prob)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9c055c2d",
   "metadata": {},
   "outputs": [],
   "source": [
    "roc_auc_score(ytest, ytest_prob)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "828933ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "fpr, tpr, thresholds = roc_curve(ytrain, ytrain_prob)\n",
    "\n",
    "# plot the ROC curve\n",
    "plt.plot(fpr, tpr)\n",
    "\n",
    "# set limits for x and y axes\n",
    "plt.xlim([0.0, 1.0])\n",
    "plt.ylim([0.0, 1.0])\n",
    "\n",
    "# plot the straight line showing worst prediction for the model\n",
    "plt.plot([0, 1], [0, 1],'r--')\n",
    "\n",
    "# add plot and axes labels\n",
    "# set text size using 'fontsize'\n",
    "plt.title('ROC curve for fraud transaction (train data set)', fontsize = 15)\n",
    "plt.xlabel('False positive rate (1-Specificity)', fontsize = 15)\n",
    "plt.ylabel('True positive rate (Sensitivity)', fontsize = 15)\n",
    "\n",
    "# add the AUC score to the plot\n",
    "# 'x' and 'y' gives position of the text\n",
    "# 's' is the text \n",
    "# use round() to round-off the AUC score upto 4 digits\n",
    "plt.text(x = 0.02, y = 0.9, s = ('AUC Score:', round(roc_auc_score(ytrain, ytrain_prob),4)))\n",
    "                               \n",
    "# plot the grid\n",
    "plt.grid(True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bad87160",
   "metadata": {},
   "outputs": [],
   "source": [
    "fpr, tpr, thresholds = roc_curve(ytest, ytest_prob)\n",
    "\n",
    "# plot the ROC curve\n",
    "plt.plot(fpr, tpr)\n",
    "\n",
    "# set limits for x and y axes\n",
    "plt.xlim([0.0, 1.0])\n",
    "plt.ylim([0.0, 1.0])\n",
    "\n",
    "# plot the straight line showing worst prediction for the model\n",
    "plt.plot([0, 1], [0, 1],'r--')\n",
    "\n",
    "# add plot and axes labels\n",
    "# set text size using 'fontsize'\n",
    "plt.title('ROC curve for fraud transaction (test dataset)', fontsize = 15)\n",
    "plt.xlabel('False positive rate (1-Specificity)', fontsize = 15)\n",
    "plt.ylabel('True positive rate (Sensitivity)', fontsize = 15)\n",
    "\n",
    "# add the AUC score to the plot\n",
    "# 'x' and 'y' gives position of the text\n",
    "# 's' is the text \n",
    "# use round() to round-off the AUC score upto 4 digits\n",
    "plt.text(x = 0.02, y = 0.9, s = ('AUC Score:', round(roc_auc_score(ytest, ytest_prob),4)))\n",
    "                               \n",
    "# plot the grid\n",
    "plt.grid(True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e31cadfc",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix,classification_report,cohen_kappa_score\n",
    "from matplotlib.colors import ListedColormap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3c5834ee",
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "cm = confusion_matrix(ytrain, ypred_train)\n",
    "\n",
    "# label the confusion matrix  \n",
    "# pass the matrix as 'data'\n",
    "# pass the required column names to the parameter, 'columns'\n",
    "# pass the required row names to the parameter, 'index'\n",
    "conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1'], index = ['Actual:0','Actual:1'])\n",
    "\n",
    "# plot a heatmap to visualize the confusion matrix\n",
    "# 'annot' prints the value of each grid \n",
    "# 'fmt = d' returns the integer value in each grid\n",
    "# 'cmap' assigns color to each grid\n",
    "# as we do not require different colors for each grid in the heatmap,\n",
    "# use 'ListedColormap' to assign the specified color to the grid\n",
    "# 'cbar = False' will not return the color bar to the right side of the heatmap\n",
    "# 'linewidths' assigns the width to the line that divides each grid\n",
    "# 'annot_kws = {'size':25})' assigns the font size of the annotated text \n",
    "sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = ListedColormap(['lightskyblue']), cbar = False, \n",
    "            linewidths = 0.1, annot_kws = {'size':25})\n",
    "\n",
    "# set the font size of x-axis ticks using 'fontsize'\n",
    "plt.xticks(fontsize = 20)\n",
    "\n",
    "# set the font size of y-axis ticks using 'fontsize'\n",
    "plt.yticks(fontsize = 20)\n",
    "\n",
    "# display the plot\n",
    "plt.title('Confusion matrix for train dataset')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e7870b62",
   "metadata": {},
   "outputs": [],
   "source": [
    "cm = confusion_matrix(ytest, ypred_test)\n",
    "\n",
    "# label the confusion matrix  \n",
    "# pass the matrix as 'data'\n",
    "# pass the required column names to the parameter, 'columns'\n",
    "# pass the required row names to the parameter, 'index'\n",
    "conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1'], index = ['Actual:0','Actual:1'])\n",
    "\n",
    "# plot a heatmap to visualize the confusion matrix\n",
    "# 'annot' prints the value of each grid \n",
    "# 'fmt = d' returns the integer value in each grid\n",
    "# 'cmap' assigns color to each grid\n",
    "# as we do not require different colors for each grid in the heatmap,\n",
    "# use 'ListedColormap' to assign the specified color to the grid\n",
    "# 'cbar = False' will not return the color bar to the right side of the heatmap\n",
    "# 'linewidths' assigns the width to the line that divides each grid\n",
    "# 'annot_kws = {'size':25})' assigns the font size of the annotated text \n",
    "sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = ListedColormap(['lightskyblue']), cbar = False, \n",
    "            linewidths = 0.1, annot_kws = {'size':25})\n",
    "\n",
    "# set the font size of x-axis ticks using 'fontsize'\n",
    "plt.xticks(fontsize = 20)\n",
    "\n",
    "# set the font size of y-axis ticks using 'fontsize'\n",
    "plt.yticks(fontsize = 20)\n",
    "\n",
    "# display the plot\n",
    "plt.title('Confusion matrix for test dataset')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c5e155dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "acc_table = classification_report(ytrain, ypred_train)\n",
    "\n",
    "# print the table\n",
    "\n",
    "print('Train Dataset Classification report:')\n",
    "print(acc_table)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7f9041f5",
   "metadata": {},
   "outputs": [],
   "source": [
    "acc_table = classification_report(ytest, ypred_test)\n",
    "\n",
    "# print the table\n",
    "print('Test Dataset Classification report:')\n",
    "print(acc_table)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9f82bf43",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_odds = pd.DataFrame(np.exp(logreg.params), columns= ['Odds']) \n",
    "\n",
    "# print the dataframe\n",
    "df_odds.sort_values('Odds',ascending=False).head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c10d25b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_odds.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6851e3f5",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_odds.index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "90ccbaaa",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in df_odds.values:\n",
    "    for j in df_odds.index:\n",
    "        \n",
    "        print(\n",
    "'odds_',j,' =', i,' it implies that the odds of detecting a fraud transaction  increases by a factor of ',i,' due to one unit increase in the', j,' keeping other variables constant'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "25259a4c",
   "metadata": {},
   "outputs": [],
   "source": [
    "kappa = cohen_kappa_score(y, ypred)\n",
    "\n",
    "# print the kappa value\n",
    "print('kappa value:',kappa)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1e51bd81",
   "metadata": {},
   "outputs": [],
   "source": [
    "print('AIC:', logreg.aic)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a4b45be8",
   "metadata": {},
   "source": [
    "# xgboost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d738cfee",
   "metadata": {},
   "outputs": [],
   "source": [
    "from xgboost import XGBClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ddc3913a",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train=df_final\n",
    "ytrain=df_target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0291a10d",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_tests=dft_final\n",
    "ytest=dft_target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2f3a9d7f",
   "metadata": {},
   "outputs": [],
   "source": [
    "xg=XGBClassifier(max_depth=200,gamma=0.2,learning_rate=0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ae4efa33",
   "metadata": {},
   "outputs": [],
   "source": [
    "model=xg.fit(x_train,ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "09d28b0e",
   "metadata": {},
   "outputs": [],
   "source": [
    "ytrain_pred=model.predict(x_train)\n",
    "ytest_pred=model.predict(x_tests)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c80c858c",
   "metadata": {},
   "outputs": [],
   "source": [
    "acc_table = classification_report(ytrain, ytrain_pred)\n",
    "\n",
    "# print the table\n",
    "\n",
    "print('Train Dataset Classification report:')\n",
    "print(acc_table)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "da0eac22",
   "metadata": {},
   "outputs": [],
   "source": [
    "roc_auc_score(ytrain,ytrain_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "41767982",
   "metadata": {},
   "outputs": [],
   "source": [
    "acc_table = classification_report(ytest, ytest_pred)\n",
    "\n",
    "# print the table\n",
    "\n",
    "print('Test Dataset Classification report:')\n",
    "print(acc_table)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "481a4871",
   "metadata": {},
   "outputs": [],
   "source": [
    "roc_auc_score(ytest,ytest_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b79476de",
   "metadata": {},
   "outputs": [],
   "source": [
    "#confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9bfa97d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "cm = confusion_matrix(ytrain, ytrain_pred)\n",
    "\n",
    "conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1'], index = ['Actual:0','Actual:1'])\n",
    "\n",
    "sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = ListedColormap(['lightskyblue']), cbar = False, \n",
    "            linewidths = 0.1, annot_kws = {'size':25})\n",
    "\n",
    "plt.xticks(fontsize = 20)\n",
    "plt.yticks(fontsize = 20)\n",
    "plt.title('Confusion matrix for train dataset')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cb0139b6",
   "metadata": {},
   "outputs": [],
   "source": [
    "cm = confusion_matrix(ytest, ytest_pred)\n",
    "conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1'], index = ['Actual:0','Actual:1'])\n",
    "\n",
    "sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = ListedColormap(['lightskyblue']), cbar = False, \n",
    "            linewidths = 0.1, annot_kws = {'size':25})\n",
    "\n",
    "plt.xticks(fontsize = 20)\n",
    "plt.yticks(fontsize = 20)\n",
    "plt.title('Confusion matrix for test dataset')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "14e75c8d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import  KFold,cross_val_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bce70853",
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "#bias error, variance error\n",
    "kf = KFold(n_splits = 10, shuffle = True, random_state = 0)\n",
    "scores = cross_val_score(xg,x_train,ytrain, cv = kf, scoring = 'roc_auc')\n",
    "\n",
    "print('Bias Error:',1-np.mean(scores))\n",
    "print('Variance Error:',np.std(scores, ddof = 1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9476ad36",
   "metadata": {},
   "outputs": [],
   "source": [
    "#roc_curve\n",
    "from sklearn.metrics import roc_auc_score,roc_curve"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1796c92f",
   "metadata": {},
   "outputs": [],
   "source": [
    "fpr, tpr, thresholds = roc_curve(ytrain, ytrain_pred)\n",
    "plt.plot(fpr, tpr)\n",
    "plt.xlim([0.0, 1.0])\n",
    "plt.ylim([0.0, 1.0])\n",
    "plt.plot([0, 1], [0, 1],'r--')\n",
    "plt.title('ROC curve for fraud transaction (train data set)', fontsize = 15)\n",
    "plt.xlabel('False positive rate (1-Specificity)', fontsize = 15)\n",
    "plt.ylabel('True positive rate (Sensitivity)', fontsize = 15)\n",
    "\n",
    "plt.text(x = 0.02, y = 0.9, s = ('AUC Score:', round(roc_auc_score(ytrain, ytrain_pred),4)))\n",
    "plt.grid(True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7bf291cb",
   "metadata": {},
   "outputs": [],
   "source": [
    "fpr, tpr, thresholds = roc_curve(ytest, ytest_pred)\n",
    "plt.plot(fpr, tpr)\n",
    "plt.xlim([0.0, 1.0])\n",
    "plt.ylim([0.0, 1.0])\n",
    "plt.plot([0, 1], [0, 1],'r--')\n",
    "plt.title('ROC curve for fraud transaction (test dataset)', fontsize = 15)\n",
    "plt.xlabel('False positive rate (1-Specificity)', fontsize = 15)\n",
    "plt.ylabel('True positive rate (Sensitivity)', fontsize = 15)\n",
    "plt.text(x = 0.02, y = 0.9, s = ('AUC Score:', round(roc_auc_score(ytest, ytest_pred),4)))\n",
    "plt.grid(True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c339f7dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(15,8))\n",
    "important_features = pd.DataFrame({'Features': x_train.columns,\n",
    "                                  'Importance':model.feature_importances_})\n",
    "important_features = important_features.sort_values('Importance', ascending = False)\n",
    "sns.barplot(x = 'Importance', y = 'Features', data = important_features)\n",
    "plt.title('Feature Importance', fontsize = 15)\n",
    "plt.xlabel('Importance', fontsize = 15)\n",
    "plt.ylabel('Features', fontsize = 15)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "65480a73",
   "metadata": {},
   "outputs": [],
   "source": [
    "important_features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9c0fcadb",
   "metadata": {},
   "outputs": [],
   "source": [
    "########################33"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "65acea4c",
   "metadata": {},
   "source": [
    "# Gradient boost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6ffa573d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import GradientBoostingClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "403fac88",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train=df_final\n",
    "ytrain=df_target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "37f4718e",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_tests=dft_final\n",
    "ytest=dft_target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e236a0e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "gb=GradientBoostingClassifier(max_depth=200,learning_rate=0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "789180e1",
   "metadata": {},
   "outputs": [],
   "source": [
    "model1=gb.fit(x_train,ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "51223d62",
   "metadata": {},
   "outputs": [],
   "source": [
    "ytrain_pred1=model1.predict(x_train)\n",
    "ytest_pred1=model1.predict(x_tests)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1ec85841",
   "metadata": {},
   "outputs": [],
   "source": [
    "acc_table = classification_report(ytrain, ytrain_pred1)\n",
    "\n",
    "# print the table\n",
    "\n",
    "print('Train Dataset Classification report:')\n",
    "print(acc_table)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f024b064",
   "metadata": {},
   "outputs": [],
   "source": [
    "acc_table = classification_report(ytest, ytest_pred1)\n",
    "\n",
    "# print the table\n",
    "\n",
    "print('Test Dataset Classification report:')\n",
    "print(acc_table)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "36133abc",
   "metadata": {},
   "outputs": [],
   "source": [
    "#confusion matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d04c3114",
   "metadata": {},
   "outputs": [],
   "source": [
    "cm = confusion_matrix(ytrain, ytrain_pred1)\n",
    "\n",
    "conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1'], index = ['Actual:0','Actual:1'])\n",
    "\n",
    "sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = ListedColormap(['lightskyblue']), cbar = False, \n",
    "            linewidths = 0.1, annot_kws = {'size':25})\n",
    "\n",
    "plt.xticks(fontsize = 20)\n",
    "plt.yticks(fontsize = 20)\n",
    "plt.title('Confusion matrix for train dataset')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8dbb0e89",
   "metadata": {},
   "outputs": [],
   "source": [
    "cm = confusion_matrix(ytest, ytest_pred1)\n",
    "conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1'], index = ['Actual:0','Actual:1'])\n",
    "sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = ListedColormap(['lightskyblue']), cbar = False, \n",
    "            linewidths = 0.1, annot_kws = {'size':25})\n",
    "plt.xticks(fontsize = 20)\n",
    "plt.yticks(fontsize = 20)\n",
    "plt.title('Confusion matrix for test dataset')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bbe51ee1",
   "metadata": {},
   "outputs": [],
   "source": [
    "#bias , variance error"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a0cc4d78",
   "metadata": {},
   "outputs": [],
   "source": [
    "#roc_curve"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9b7b5d4c",
   "metadata": {},
   "outputs": [],
   "source": [
    "fpr, tpr, thresholds = roc_curve(ytrain, ytrain_pred1)\n",
    "\n",
    "plt.plot(fpr, tpr)\n",
    "\n",
    "plt.xlim([0.0, 1.0])\n",
    "plt.ylim([0.0, 1.0])\n",
    "plt.plot([0, 1], [0, 1],'r--')\n",
    "plt.title('ROC curve for fraud transaction (train data set)', fontsize = 15)\n",
    "plt.xlabel('False positive rate (1-Specificity)', fontsize = 15)\n",
    "plt.ylabel('True positive rate (Sensitivity)', fontsize = 15)\n",
    "plt.text(x = 0.02, y = 0.9, s = ('AUC Score:', round(roc_auc_score(ytrain, ytrain_pred1),4)))\n",
    "plt.grid(True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "05282818",
   "metadata": {},
   "outputs": [],
   "source": [
    "fpr, tpr, thresholds = roc_curve(ytest, ytest_pred1)\n",
    "plt.plot(fpr, tpr)\n",
    "plt.xlim([0.0, 1.0])\n",
    "plt.ylim([0.0, 1.0])\n",
    "plt.plot([0, 1], [0, 1],'r--')\n",
    "plt.title('ROC curve for fraud transaction (test dataset)', fontsize = 15)\n",
    "plt.xlabel('False positive rate (1-Specificity)', fontsize = 15)\n",
    "plt.ylabel('True positive rate (Sensitivity)', fontsize = 15)\n",
    "plt.text(x = 0.02, y = 0.9, s = ('AUC Score:', round(roc_auc_score(ytest, ytest_pred1),4)))\n",
    "plt.grid(True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "958b21ac",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8ab253d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "#########################################"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
