import os, glob  #
import numpy as np  #数値計算用モジュール
import pandas as pd  #データ処理用モジュール
import xlwings as xw  #Excel操作用モジュール
import matplotlib.pyplot as plt  #データ可視化モジュール
import seaborn as sns  #matplotlib拡張モジュール

from sklearn.model_selection import train_test_split  #トレーニングデータとテストデータの分割用メソッド
from sklearn import linear_model  #回帰分析用モジュール
from sklearn.metrics import mean_absolute_error  #平均絶対誤差(MAE)。小さいほどいい。
from sklearn.metrics import mean_squared_error  #平均二乗誤差(MSE)。小さいほどいい。


def get_csv_data(csv_relative_path):
    #xlwingsでExcelファイルを読み込み
    file_relative_path = os.path.dirname(__file__)
    
    excel_relative_path = "analysis_dummy.xlsx"
    #wb = xw.Book()  #Excel New Bookを作成
    wb = xw.Book(os.path.join(file_relative_path, excel_relative_path))  #既存のファイルを読み込み
    sht = wb.sheets[u'ダミーデータ']  #操作するシートのインスタンスを作成
    table_loc = "A1"  #Excelデータの読み込み位置(左上)

    #データフレームを作成
    obj_df_all = sht.range(table_loc).options(pd.DataFrame, expand='table').value.reset_index()  #Excelから読み込み
    print(obj_df_all)

    #データの加工
    obj_df_all = obj_df_all[['最長長さ', '中間長さ', '最小長さ', '面積', '体積', '個数', '単価']].astype(np.float64)
    #obj_df_all = obj_df_all[['最長長さ', '中間長さ', '最小長さ', '面積', '体積', '単価']].astype(np.float64)
    #obj_df_all = obj_df_all[['最長長さ', '中間長さ', '最小長さ', '単価']].astype(np.float64)
    #obj_df_all = obj_df_all[['面積', '体積', '単価']].astype(np.float64)

    return obj_df_all


def show_pairplot(obj_df_all):
    #散布図、ヒストグラム、ピアソンの積率相関係数
    #sns.jointplot('面積', '単価', data=obj_df_all, kind = "reg")
    #plt.show()

    #回帰直線を引いた全変数間の散布図とヒストグラム
    sns.pairplot(obj_df_all, kind = "reg", vars = ['最長長さ', '中間長さ', '最小長さ', '単価'])
    #sns.pairplot(obj_df_all, kind = "reg", vars = ['面積', '体積', '個数', '単価'])
    plt.show()

    #sns.distplot(obj_df_all['面積'])
    #sns.lmplot(x='個数',y='単価', data=obj_df_all)


def show_predict(clf, train_x, train_y, test_x, test_y):
    plt.plot(clf.predict(test_x), test_y, "o")  #予測値と実績値の散布図
    plt.plot(test_y, test_y)  #理想の直線
    #plt.title("Predicted-Real Gpaph")  #グラフタイトル
    plt.xlabel("予測単価")  #x軸ラベル
    plt.ylabel("実績単価")  #y軸ラベル
    plt.show()  #グラフ表示

    #残差プロット
    plt.axhline(0, color='#000000')  #f(x)=0の線
    plt.scatter(clf.predict(train_x), clf.predict(train_x) -train_y, marker='o', label='Train Data')
    plt.scatter(clf.predict(test_x), clf.predict(test_x) -test_y, marker='s', label='Test Data')
    plt.legend(loc='best') #凡例(グラフラベル)を表示
    plt.xlabel("予測値")  #x軸ラベル
    plt.ylabel("残差")  #y軸ラベル
    plt.show()  #グラフ表示



if (__name__ == '__main__'):
    csv_relative_path = r"analysis_dummy.csv"
    obj_df_all = get_csv_data(csv_relative_path)

    show_pairplot(obj_df_all)


    #データセットを説明変数と目的変数に分割
    obj_x = obj_df_all.drop("単価", axis=1).values  #目的変数以外を抜き出し、説明変数を設定
    obj_y = obj_df_all['単価'].values  #目的変数を設定
    print(obj_x)
    print(obj_y)
    print()


    #サンプリングを実施し、トレーニングデータ、テストデータに分割
    train_x, test_x, train_y, test_y = train_test_split(obj_x, obj_y, test_size=0.3)
    '''print(train_x)
    print(test_x)
    print(train_y)
    print(test_y)
    print()'''
    #clf = analyze_corr(obj_df_all, train_x, train_y)

    #回帰分析用クラスを用意
    clf = linear_model.LinearRegression()

    clf.fit(train_x, train_y)  #線形予測モデルを作成

    print("切片(誤差)：",clf.intercept_)
    print("偏回帰係数：\n",pd.DataFrame({"Name":obj_df_all.drop("単価", axis=1).columns, "Coef":clf.coef_}) )
    print("学習用データでの決定係数R^2：",clf.score(train_x, train_y))
    print("検証用データでの決定係数R^2：",clf.score(test_x, test_y))

    #平均絶対誤差。実際の値と予測値の絶対値の平均
    print("学習用データでの平均絶対誤差：",mean_absolute_error(clf.predict(train_x), train_y))
    print("学習用データでの平均絶対誤差：",mean_absolute_error(clf.predict(test_x), test_y))
    print()

    show_predict(clf, train_x, train_y, test_x, test_y)
