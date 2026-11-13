## Dữ liệu, cách chia dữ liệu và các mô hình cơ sở

### Dữ liệu
Đồ án dùng dữ liệu doanh số bán lẻ của Walmart ở Mỹ trong cuộc thi "M5 Forecasting - Accuracy" trên Kaggle: 1.941 ngày, từ 29/01/2011 đến 22/05/2016. Từ đó lấy một tập con gồm 297 chuỗi (mỗi chuỗi là doanh số theo ngày của một sản phẩm ở một cửa hàng): nhóm hàng FOODS, ba cửa hàng CA_3, TX_2, WI_1 (mỗi bang một cửa hàng bán mạnh nhất), mỗi cửa hàng 99 sản phẩm chọn ngẫu nhiên (hạt giống ngẫu nhiên 29) chia đều ba mức bán thấp, vừa, cao. Chỉ giữ sản phẩm đã bán ít nhất 1.000 ngày kể từ lần bán đầu tiên, để có đủ lịch sử.

Một số đặc điểm của tập con: 47,5% số ngày bán 0 (mỗi chuỗi từ 4% đến 93%); khi có bán, số lượng thường nhỏ (trung vị 2 đơn vị); mùa vụ theo tuần rõ (thứ Bảy và Chủ Nhật cao nhất); ngày 25/12 doanh số bằng 0 ở mọi chuỗi. Hai giả định quan trọng: doanh số được coi là nhu cầu (vì M5 không có số tồn kho, nên không phân biệt được ngày bán 0 do không ai mua hay do hết hàng), và ngày 25/12 là ngày cửa hàng đóng cửa.

### Cách chia dữ liệu
Dữ liệu được chia theo thời gian, cùng một mốc cho mọi chuỗi:

| Tập | Từ | Đến | Số ngày |
|---|---|---|---|
| Huấn luyện | 29/01/2011 | 21/06/2015 | 1.605 |
| Kiểm định (validation) | 22/06/2015 | 06/12/2015 | 168 |
| Kiểm tra (test) | 07/12/2015 | 22/05/2016 | 168 |

Tập kiểm định và tập kiểm tra mỗi tập chia thành 6 cửa sổ liên tiếp dài 28 ngày. Với mỗi cửa sổ, mô hình đứng ở cuối ngày ngay trước cửa sổ, chỉ dùng số liệu thật đến lúc đó để dự báo 28 ngày kế tiếp; sau đó số liệu thật của cửa sổ này được dùng làm quá khứ cho cửa sổ tiếp theo. Độ dài 28 ngày là giả định, lấy theo cuộc thi M5. Mọi lựa chọn (độ dài cửa sổ trung bình,...) được thực hiện trên tập kiểm định; tập kiểm tra chỉ chạy một lần với cấu hình đã khóa. Ngày 25/12 bị loại khỏi các phép đo.

### Thước đo
Mỗi thước đo được tính riêng cho từng chuỗi, sau đó lấy trung bình và trung vị qua 297 chuỗi (để một chuỗi bán rất nhiều không lấn át các chuỗi khác).
- **RMSSE** (thước chính): sai số bình phương trung bình của dự báo, chia cho mức dao động ngày-qua-ngày của chính chuỗi trong quá khứ, rồi lấy căn. Vì đã chia theo quy mô của từng chuỗi nên so sánh được giữa chuỗi bán nhiều và bán ít; càng thấp càng tốt. Đây là thành phần của WRMSSE, thước đo chính của cuộc thi M5, nhưng điểm của đồ án không so được với bảng xếp hạng M5 vì ta không tính đầy đủ các mức tổng hợp và trọng số.
- **MAE** và **RMSE**: sai số trung bình theo đơn vị (số lượng bán mỗi ngày).
- **sMAPE**: sai số theo phần trăm, chỉ là thước phụ. Với chuỗi nhiều ngày bán 0, nó xếp hạng ngược với các thước còn lại, nên không dùng để chọn mô hình.

### Các mô hình cơ sở (baseline)
Baseline là các cách dự báo đơn giản dùng làm mốc so sánh. Bảy cách được thử:
1. Toàn số 0 (chỉ để tham chiếu).
2. Naive: mọi ngày tương lai bằng số bán của ngày cuối cùng.
3. Seasonal naive: lặp lại 7 ngày cuối.
4. Trung bình trượt: trung bình các ngày gần nhất (chọn 28 ngày trên tập kiểm định).
5. Trung bình theo thứ: mỗi thứ trong tuần bằng trung bình của cùng thứ đó trong các tuần gần nhất (chọn 16 tuần).
6. Croston (biến thể SBA): cách dành cho chuỗi thưa, dự báo một mức phẳng.
7. ETS (làm mượt theo trọng số giảm dần, thư viện statsforecast tự chọn kiểu mô hình, chu kỳ mùa vụ 7 ngày).

### Kết quả trên tập kiểm tra

| Cách | RMSSE (trung bình / trung vị) | MAE (trung bình / trung vị) |
|---|---|---|
| Toàn số 0 | 0,963 / 0,924 | 2,378 / 1,131 |
| Naive | 0,954 / 0,916 | 2,086 / 1,384 |
| Seasonal naive | 0,944 / 0,931 | 1,890 / 1,306 |
| Trung bình trượt 28 ngày | 0,728 / 0,698 | 1,644 / 1,059 |
| Trung bình theo thứ 16 tuần | 0,754 / 0,714 | 1,664 / 1,087 |
| Croston (SBA) | 0,761 / 0,723 | 1,734 / 1,111 |
| **ETS** | **0,721 / 0,691** | **1,557 / 1,047** |

Nhận xét:
- ETS và trung bình trượt 28 ngày là hai cách tốt nhất, và hơn hẳn naive, seasonal naive và toàn số 0. Thứ hạng của bốn cách đầu giống nhau ở tập kiểm định và tập kiểm tra.
- Khoảng cách giữa ETS và trung bình trượt nhỏ (RMSSE 0,721 so với 0,728) và chưa được đo độ chắc chắn. MAE trung bình của ETS thấp hơn 5,3%, nhưng khoảng một nửa lợi thế này đến từ một chuỗi bán rất lớn; bỏ chuỗi đó ra còn khoảng 2,9%. Vì thế khi so sánh nên nhìn RMSSE trước.
- Mọi cách đều kém hơn trên tập kiểm tra so với tập kiểm định (RMSSE trung bình tăng từ 0,036 đến 0,083); các cách không chỉnh gì cũng tệ đi, nên phần lớn chênh lệch không do chọn cấu hình trên tập kiểm định. Giai đoạn kiểm tra có thể khó dự báo hơn (giả định, chưa tìm nguyên nhân).

### Hạn chế
- Mỗi lần dự báo 28 ngày một lần; mô phỏng kho ở các tuần sau có thể cần cập nhật dự báo thường xuyên hơn.
- Dự báo hiện là một con số cho mỗi ngày, chưa phải khoảng; đây là việc của các tuần sau.
- ETS và Croston vẫn thấy số 0 của ngày 25/12 trong lịch sử (ảnh hưởng chắc nhỏ nhưng chưa kiểm chứng); ETS có khoảng 21–24% số chuỗi có ít nhất một dự báo âm (đã đặt về 0).
- Tập kiểm tra chỉ gồm 6 cửa sổ trong một giai đoạn, và các chuỗi cùng cửa hàng nên không hoàn toàn độc lập.
- Tập con chỉ gồm cửa hàng bán mạnh, nhóm hàng FOODS và sản phẩm đã bán lâu; dữ liệu là của Walmart ở Mỹ giai đoạn 2011 đến 2016, nên kết luận chưa chắc đúng cho cửa hàng bán chậm, sản phẩm mới hay hành vi mua sắm hiện nay.