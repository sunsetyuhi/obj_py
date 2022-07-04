import glob  #ファイル操作用モジュール
import numpy as np  #数値計算用モジュール
import pandas as pd  #データ処理用モジュール
import xlwings as xw  #Excel操作用モジュール



#フォルダ内のファイル名を取得
obj_names = glob.glob(u'./obj_files/*.obj')  #正規表現で検索
print ("obj_names =", obj_names)
print()

#xlwingsでExcelファイルを読み込み
excel_name = "import_obj.xlsm"
#wb = xw.Book()  #Excel New Bookを作成
wb = xw.Book(excel_name)  #既存のファイルを読み込み
sht = wb.sheets[u'一覧表']  #操作するシートのインスタンスを作成
table_loc = "C5"  #Excelデータの読み込み位置(左上)



#列ラベル
obj_label01 = ['製品名','部品名','最長長さ','中間長さ','最小長さ']
obj_label02 = ['面積','体積','頂点数','試作単価']

#データフレームを作成
#obj_df_all = pd.DataFrame(index=[], columns=obj_label01+obj_label02)  #空のデータを作成
obj_df_all = sht.range(table_loc).options(pd.DataFrame, expand='table').value  #Excelから読み込み
print(obj_df_all)



#各objファイルのx,y,z長さを計算
for i in range(len(obj_names)):
  obj_name = obj_names[i][12:-4].split("_",1)  #不要文字を削除し、最初の_で区切る
  print("name =", obj_name)

  #objデータから頂点データを取得
  obj_vtx = pd.read_csv(obj_names[i], encoding="shift_jis", skiprows=2, header=None, sep='\s+')
  obj_vtx.columns = ['data', 'x', 'y', 'z']  #列ラベルをつける
  obj_vtx = obj_vtx[obj_vtx['data']=='v']  #頂点データのみ取得して置換
  obj_vtx = obj_vtx[['x', 'y', 'z']].astype(np.float64)  #float64に型変換
  #print(obj_vtx.head(5))

  #x,y,z方向の長さ
  obj_len = np.empty(3, np.float)
  obj_len[0] = max(obj_vtx['x']) -min(obj_vtx['x'])
  obj_len[1] = max(obj_vtx['y']) -min(obj_vtx['y'])
  obj_len[2] = max(obj_vtx['z']) -min(obj_vtx['z'])
  obj_len = np.sort(obj_len)[::-1]  #降順(大きい順)にソート
  print("xl =",obj_len[0], ",   yl =",obj_len[1], ",   zl =",obj_len[2])

  #最外形の面積・体積
  obj_area = obj_len[0]*obj_len[1]
  obj_vol = obj_len[0]*obj_len[1]*obj_len[2]
  print("area = ",obj_area, ",   vol =",obj_vol)

  #頂点数
  obj_vtx_total = len(obj_vtx['x'])
  print("vtx total = ",obj_vtx_total)

  #データフレームに追加
  obj_se01 = pd.Series([obj_name[0],obj_name[1], obj_len[0],obj_len[1],obj_len[2]], index=obj_label01)
  obj_se02 = pd.Series([obj_area,obj_vol, obj_vtx_total, ''], index=obj_label02)
  obj_se_all = pd.concat([obj_se01,obj_se02])  #データを横に結合
  obj_df_all = obj_df_all.append(obj_se_all, ignore_index=True)  #データを縦に結合
  print()


#データの体裁を整える
obj_df_all = obj_df_all.drop_duplicates(subset=['製品名','部品名'], keep='first')  #重複してたら後ろを消す
obj_df_all = obj_df_all.sort_values(['製品名','部品名'], ascending=[True, True])  #昇順(小さい順)にソート

print(obj_df_all)



#pandasでExcelに書き出し
'''excel_writer = pd.ExcelWriter('01_モデルデータ.xlsx')  #出力ファイル名を指定
obj_df_all.to_excel(excel_writer, '一覧表', index=False)  #シート名を指定してデータフレームを書き出す
excel_writer.save()  #書き出した内容を保存'''


#xlwingsでExcelに書き出し
sht.range(table_loc).value = obj_df_all  #Excelにデータフレームを書き込み
wb.save(excel_name)  #保存'''
