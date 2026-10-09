"""小助理自己訓練的小模型(傳統機器學習,照課程教材的寫法):

- 決策樹(id3.py,照課程 DecisionTree.py 的 ID3):判斷這題「一般 / 難題」,學 Super 分類器的判斷;
  學得夠像就能取代它(不花 token、不用等、規則看得懂)。
- 邏輯回歸(regression.py):做同一件事,拿來跟決策樹比較。
- 線性回歸(regression.py):回答前預估這一輪大概花多少 tokens、幾秒。

訓練:train_models.py(會用到 numpy / scipy / scikit-learn / matplotlib,只有訓練和實驗需要)。
執行:runtime.py 讀訓練好的 JSON(樹的節點、回歸係數),純 Python 計算,不載入那些套件,也不用 pickle。
"""
