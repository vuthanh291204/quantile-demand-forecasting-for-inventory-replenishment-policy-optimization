# Giả định và quyết định — Đồ án 3 (cập nhật cuối Tuần 1)

Cách đọc các nhãn trong file này:
- [Đã kiểm chứng]: đã chạy code trên dữ liệu và thấy đúng (ghi rõ ở notebook nào).
- [Giả định]: tự chọn hoặc tự đoán, chưa có nguồn xác nhận.
- [Chưa biết]: chưa có thông tin, cần tìm nguồn hoặc quyết định sau.
- [Chờ xác nhận]: đề xuất tạm, quyết định sau.

---

## 1. Dữ liệu

| Mục | Nội dung | Nhãn |
|---|---|---|
| Nguồn | Cuộc thi "M5 Forecasting - Accuracy" trên Kaggle (dữ liệu bán lẻ của Walmart ở Mỹ). Giấy phép ghi trên trang: "Subject to Competition Rules". Đã đồng ý điều khoản bằng nút "Late Submission". | Đã kiểm chứng (trang Kaggle) |
| Dung lượng | Khoảng 450 MB khi nén, 5 file | Đã kiểm chứng (trang Kaggle) |
| File dùng | `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv` | Quyết định |
| File không dùng | `sample_submission.csv` (chỉ để nộp bài thi); `sales_train_validation.csv` (là phần đầu của bản evaluation) | Đã kiểm chứng (notebook 01) |
| Khoảng thời gian | d_1 đến d_1941, tức 2011-01-29 đến 2016-05-22. Năm 2016 chỉ có 143 ngày. | Đã kiểm chứng (notebook 01, 02) |
| Số chuỗi | 30.490 chuỗi = 3.049 sản phẩm × 10 cửa hàng; 3 bang (CA, TX, WI); 3 nhóm hàng (FOODS, HOBBIES, HOUSEHOLD); 7 bộ phận | Đã kiểm chứng (notebook 01) |
| Cách nối bảng | Cột ngày `d_x` của doanh số ↔ cột `d` của lịch. Lịch có `wm_yr_wk` (mã tuần) nối với bảng giá theo `store_id + item_id + wm_yr_wk`; khóa này không bị trùng. | Đã kiểm chứng (notebook 01) |
| Bảng lịch có 1.969 dòng, nhiều hơn 1.941 ngày doanh số 28 dòng | 28 ngày cuối của giai đoạn cần dự báo trong cuộc thi, chưa có doanh số | Đã kiểm chứng (trang Kaggle) |
| Ý nghĩa `snap_CA/TX/WI` | Cờ 0/1: ngày bang đó cho dùng trợ cấp thực phẩm SNAP của chính phủ Mỹ | Đã kiểm chứng (trang Kaggle) |
| Ý nghĩa `wday` | Số thứ tự của thứ trong tuần, bắt đầu từ thứ Bảy | Đã kiểm chứng (đối chiếu từ cột `weekday`) |
| Đơn vị tiền của `sell_price` | Đô la Mỹ (USD) | Đã kiểm chứng |

---

## 2. Tập con dùng cho Đồ án 3

| Tham số | Giá trị | Lý do |
|---|---|---|
| Nhóm hàng | FOODS | Có số bán trung bình cao nhất và tỉ lệ ngày bán 0 thấp nhất trong 3 nhóm (trung vị trên các chuỗi của nhóm: 0,71 đơn vị/ngày (tính cả ngày chưa bán); 54% ngày bán 0 (tính từ lần bán đầu tiên)) |
| Cửa hàng | CA_3, TX_2, WI_1 | Mỗi bang một cửa hàng, là cửa hàng có số bán trung bình cao nhất trong bang. Số này tính trên cả 3 nhóm hàng, không riêng FOODS. |
| Ngưỡng lịch sử | Chuỗi phải có ít nhất 1.000 ngày kể từ lần bán đầu tiên | Giả định: cần đủ ngày để chia train/validation/test và thấy mùa vụ. Còn lại 3.584 trong 4.311 chuỗi FOODS ở 3 cửa hàng (ngưỡng 500: 4.236; 1.500: 2.472; 1.800: 2.056). |
| Số chuỗi | 297 = 3 cửa hàng × 3 mức bán × 33 chuỗi | Nằm trong khoảng 100–500 của kế hoạch; đủ nhỏ để chạy mô phỏng nhiều lần |
| Cách chọn | Chia 3 mức bán (thấp/vừa/cao) riêng trong từng cửa hàng, lấy ngẫu nhiên 33 chuỗi mỗi nhóm | Để tập con có đủ loại chuỗi, không lệch về hàng bán chạy |
| Mức bán dùng để chia | Trung bình số bán mỗi ngày kể từ lần bán đầu tiên | Tránh coi sản phẩm mới ra mắt là bán ít |
| SEED | 29 | Số bất kỳ, chọn một lần và giữ nguyên để chạy lại ra đúng kết quả |

File danh sách chuỗi: `configs/subset_ids.csv` (đưa lên Git). Dữ liệu đã lọc nằm trong `data/processed/` (không đưa lên Git); tạo lại bằng notebook `01_kham_pha_bang.ipynb` rồi `02_kham_pha_tap_con.ipynb`.

Lưu ý: nhóm mức bán 0/1/2 là tương đối trong từng cửa hàng, nên "nhóm 2" của hai cửa hàng khác nhau không bán nhiều bằng nhau.

Hạn chế của tập con:
- Chỉ gồm cửa hàng bán mạnh; kết luận chưa chắc đúng cho cửa hàng bán chậm.
- Chỉ gồm sản phẩm đã bán ít nhất 1.000 ngày, nên chưa chắc đại diện cho sản phẩm mới ra mắt.
- Chỉ một nhóm hàng (FOODS).
- Dữ liệu của Walmart ở Mỹ, giai đoạn 2011–2016; hành vi mua sắm hiện nay có thể khác.

---

## 3. Định nghĩa "chuỗi đang bán" (`live`)

- Một chuỗi coi là đang bán từ ngày có doanh số lớn hơn 0 đầu tiên. Các ngày trước đó không tính là "nhu cầu bằng 0".
- Cơ sở [Đã kiểm chứng, notebook 02]:
  - Trong 499.718 dòng "đang bán", không có dòng nào thiếu giá.
  - Trong 76.759 dòng "chưa bán", 76.125 dòng thiếu gi  á (chỉ 634 dòng đã có giá). Vậy thiếu giá nghĩa là sản phẩm chưa được bán.
  - Giá xuất hiện sớm hơn lần bán đầu tiên từ 0 đến 6 ngày (trung bình 2,1 ngày); giá ghi theo tuần nên khoảng này là bình thường.
  - WI_1 thiếu giá nhiều hơn (22,2%, so với 9,2% ở CA_3 và 8,2% ở TX_2), khớp với tỉ lệ ngày chưa bán của cửa hàng này (22,4%).
- Hệ quả: tỉ lệ ngày bán 0 ở mục 4 chỉ tính từ lần bán đầu tiên.

---

## 4. Quan sát dữ liệu (tập con 297 chuỗi, notebook 02)

Tất cả con số dưới đây là [Đã kiểm chứng] trên tập con này; các nhận định về nguyên nhân được ghi rõ là [Giả định].

Ngày bán 0 và quy mô
- Tỉ lệ ngày bán 0 chung: 47,5%. Từng chuỗi khác nhau nhiều: từ 4% đến 93%.
- Theo mức bán (thấp/vừa/cao) ở CA_3, TX_2, WI_1: nhóm thấp 65–77% ngày bán 0, nhóm cao 20–31%.
- Khi có bán, trung vị là 2 đơn vị, 75% trường hợp không quá 5. Tính cả ngày bán 0: 90% ngày không quá 6 đơn vị, 99% không quá 31.
- Có một chuỗi bán rất nhiều: `FOODS_3_090_CA_3` (trung bình 130,8 đơn vị/ngày, lớn nhất 763). Đây là chuỗi duy nhất có trung bình trên 50.

Xu hướng
- Số bán trung bình mỗi chuỗi giảm từ 3,7 (2012) xuống 2,2 (2015), nhưng số chuỗi đang bán tăng từ 179 lên 297 nên con số này lẫn hiệu ứng chuỗi mới vào.
- Chỉ xét 94 chuỗi bán từ ngày đầu tiên: tăng từ 4,8 (2011) lên 6,1 (2012), rồi giảm xuống 3,7 (2015). Năm 2016 (3,9) chỉ có 143 ngày nên chưa so sánh công bằng.
- Nguyên nhân của đợt giảm: [Chưa biết].

Mùa vụ theo tuần (mức bán so với trung bình của chính chuỗi, 1,0 = trung bình)
- Thứ Bảy và Chủ Nhật cao nhất (khoảng 1,11–1,24); thứ Ba đến thứ Năm thấp nhất (khoảng 0,86–0,94). Hình dạng giống nhau ở cả 3 cửa hàng.

Mùa vụ theo tháng (đã tách xu hướng theo năm, chỉ dùng 2012–2015)
- Tháng 1–2 cao hơn trung bình (khoảng 1,10–1,12); các tháng khác khoảng 0,95–1,03 và không ổn định giữa các năm.
- [Giả định]: mùa vụ theo tháng yếu hơn nhiều so với mùa vụ theo tuần; phần cao ở đầu năm có thể lẫn với xu hướng giảm ngay trong từng năm, chưa tách được.

Sự kiện và SNAP
- Ngày SNAP bán cao hơn khoảng 13% (1,086 so với 0,958), đúng chiều ở cả 3 cửa hàng.
- Giáng sinh (25/12): doanh số bằng 0 ở mọi chuỗi trong các năm 2012–2015, năm 2011 chỉ có 1 đơn vị. [Giả định]: cửa hàng đóng cửa. Nếu đúng thì đây không phải nhu cầu bằng 0.
- Các ngày lễ khác lệch rõ so với trung bình (ví dụ Lễ Tạ ơn thấp, Super Bowl cao), nhưng mỗi sự kiện chỉ có vài ngày thật nên mức trung bình từng sự kiện chưa chắc chắn.

Giá
- Giá đổi ít: 15% chuỗi không bao giờ đổi giá; trung vị mỗi chuỗi có 4 mức giá và 4 lần đổi giá. Mỗi lần đổi, trung vị khoảng 10%, 75% lần đổi không quá 19%.
- Giá thường lệ tăng nhẹ theo năm (tỉ lệ so với trung vị của chuỗi: 0,97 năm 2011 lên 1,02 năm 2015).
- Khi giá cao hơn thường lệ 5–20%, doanh số thấp hơn so với khi giá bình thường, đúng ở cả 6 năm. Khi giá thấp hơn 5–20%, kết quả không nhất quán giữa các năm. Các nhóm giá lệch rất mạnh có ít quan sát và kết quả thất thường. Đây chỉ là mối liên hệ (giảm giá thường đi kèm khuyến mãi, sự kiện), chưa phải tác động của giá.

---

## 5. Điểm bất thường và việc cần quyết định

| Điểm | Hiện trạng | Quyết định tạm | Nhãn |
|---|---|---|---|
| `FOODS_3_090_CA_3` bán rất nhiều (nhiều ngày 600–760 đơn vị) | Lặp lại ở nhiều ngày nên không giống một lỗi nhập đơn lẻ | Giữ trong tập con, gắn cờ; khi đánh giá phải tính thước đo từng chuỗi rồi mới lấy trung bình, tránh một chuỗi lấn át | Đã quyết (tạm) |
| `FOODS_2_281_TX_2` có giá 0,33 trong một tuần (2014-03-01) giữa các mức 5,97–7,98 | Làm `max_over_min` = 24 và một lần đổi giá +1709% | Giữ; xem lại khi làm đặc trưng giá ở Tuần 3 (có thể coi là ngoại lệ) | Đã quyết (tạm) |
| Giá thấp 0,15–0,25 (`FOODS_3_547_WI_1`) | Chưa biết là mặt hàng rẻ hay dữ liệu lạ | Giữ | Chưa biết |
| Ngày Giáng sinh | Doanh số bằng 0 ở mọi chuỗi trong 2012–2015, và 1 đơn vị tổng cộng năm 2011 (đã kiểm chứng lại ở Tuần 2) | Coi là ngày cửa hàng đóng cửa; loại khỏi MAE, RMSE, sMAPE và ghi số ngày bị loại. Cách xử lý trong mô phỏng kho: quyết định ở Tuần 5 | Đã quyết (tạm); nguyên nhân đóng cửa là Giả định |
| Mốc chia train / validation / test | Đã chốt ở Tuần 2, xem mục 9 | Chia theo thời gian, cùng một mốc cho mọi chuỗi | Đã quyết |

---

## 6. Hệ quả dự kiến cho các tuần sau (cần kiểm tra, chưa phải kết luận)

- Thước đo (Tuần 2): nhiều ngày bán 0 và quy mô chuỗi rất khác nhau, nên cần thước đo không phụ thuộc quy mô và tính riêng từng chuỗi; sMAPE cần xử lý ngày bán 0.
- Chia dữ liệu và kiểm tra: nhu cầu không đứng yên theo thời gian, nên chia theo thời gian và kiểm tra lại trên nhiều giai đoạn.
- Đặc trưng (Tuần 3): thứ trong tuần là tín hiệu mạnh nhất quan sát được; thêm cờ sự kiện và SNAP. Giá có thể đưa vào như một đặc trưng nhưng bằng chứng hiện tại là yếu.
- Dự báo phân vị (Tuần 4): số bán nhỏ và nhiều ngày bằng 0 nên nhiều mức phân vị có thể trùng nhau (cùng là 0 hoặc 1). Cần kiểm tra khi huấn luyện.
- Mô phỏng kho (Tuần 5): ngày Giáng sinh cần quy ước riêng.

---

## 7. Tham số vận hành (chi phí lưu kho, thời gian chờ hàng, ...)

Chưa quyết ở Tuần 1. Đây đều là giả định, không có trong dữ liệu M5, và mỗi tham số phải ghi kèm nguồn tham khảo khi chốt:
- Chi phí lưu kho: tỉ lệ phần trăm giá trị hàng mỗi năm (Tuần 6). Nguồn: chưa có.
- Thời gian chờ hàng: dự kiến thử vài mức như 3, 7, 14 ngày (Tuần 9).
- Mức đáp ứng đơn mục tiêu: dự kiến thử vài mức như 90%, 95%, 98% (Tuần 6).
- Hàng thiếu thì mất khách hay nợ đơn: quyết định ở Tuần 5.

---

## 8. Môi trường và khả năng chạy lại

- Python 3.12.10, Windows + PowerShell, VS Code. Thư viện: `requirements.txt` (tên thư viện) và `requirements-lock.txt` (kèm số phiên bản).
- Seed ngẫu nhiên: 29, đã ghi cùng với cách chọn tập con.
- Thứ tự chạy lại: tải dữ liệu vào `data/raw/` → notebook `01_kham_pha_bang.ipynb` → notebook `02_kham_pha_tap_con.ipynb`.

## 9. Chia dữ liệu theo thời gian

Mốc chia lưu ở `configs/split.toml`, code ở `src/data/split.py`, test ở `tests/test_split.py`.

| Tập | Ngày dữ liệu | Từ | Đến | Số ngày |
|---|---|---|---|---|
| Train | d_1 – d_1605 | 2011-01-29 (thứ Bảy) | 2015-06-21 (Chủ Nhật) | 1.605 |
| Validation | d_1606 – d_1773 | 2015-06-22 (thứ Hai) | 2015-12-06 (Chủ Nhật) | 168 |
| Test | d_1774 – d_1941 | 2015-12-07 (thứ Hai) | 2016-05-22 (Chủ Nhật) | 168 |

- Validation và test mỗi tập chia thành 6 cửa sổ liên tiếp, mỗi cửa sổ 28 ngày.
- Độ dài mỗi lần dự báo (28 ngày).
- Mỗi cửa sổ dài 28 ngày nên mỗi thứ trong tuần xuất hiện đúng 4 lần. Tuần của dữ liệu M5 (`wday`, `wm_yr_wk`) chạy từ thứ Bảy đến thứ Sáu, còn cửa sổ của ta chạy từ thứ Hai đến Chủ Nhật; điều này không ảnh hưởng mùa vụ tuần. Ở Tuần 3, khi tạo đặc trưng theo tuần, chỉ dùng dữ liệu đến hết ngày cắt.
- Cả 297 chuỗi đều đã bán trước khi validation bắt đầu. Số ngày lịch sử trong train kể từ lần bán đầu tiên: nhỏ nhất 669, trung vị 1.541. [Đã kiểm chứng]
- Ngày 25/12: train có 4 ngày (2011–2014), validation không có, test có 1 ngày (d_1792).

Hạn chế:
- Validation (tháng 6–12/2015) và test (tháng 12/2015–5/2016) rơi vào các mùa khác nhau. Mùa vụ tháng yếu nên chắc ảnh hưởng nhỏ, nhưng chưa kiểm chứng.
- Test chỉ dài 168 ngày, thuộc một giai đoạn duy nhất.
- [Giả định] Doanh số được coi là nhu cầu. M5 không có số tồn kho, nên ngày bán 0 có thể là do không ai mua hoặc do hết hàng, ta không phân biệt được. Hệ quả: nhu cầu thật có thể cao hơn doanh số quan sát.

## 10. Thước đo dự báo (Tuần 2)

Code ở `src/evaluation/metrics.py`, test ở `tests/test_metrics.py`.

- Thước đo chính, không phụ thuộc quy mô: **RMSSE** (root mean squared scaled error) tính riêng từng chuỗi. Đây là thành phần từng chuỗi của WRMSSE, thước đo chính của cuộc thi M5 (nguồn: bài báo The M5 Accuracy competition: Results, findings and conclusions). Cách tính: sai số bình phương trung bình của dự báo trên cửa sổ đánh giá, chia cho trung bình bình phương mức thay đổi từ ngày này sang ngày kế tiếp trong lịch sử của chính chuỗi, rồi lấy căn. Phần lịch sử bắt đầu từ ngày bán đầu tiên của chuỗi.
- **Không dùng WRMSSE đầy đủ.** WRMSSE gộp nhiều mức tổng hợp của toàn bộ 30.490 chuỗi và có trọng số theo doanh thu; tập con 297 chuỗi và bài toán đặt hàng theo từng sản phẩm × cửa hàng không khớp với cách đó. Hệ quả: điểm RMSSE của đồ án **không so được** với bảng xếp hạng M5. Chưa dùng trọng số doanh thu. [Quyết định, xem lại nếu cần]
- Thước đo đọc theo đơn vị: **MAE**, **RMSE** (số lượng bán mỗi ngày). Thước phụ: **sMAPE** (phần trăm, 0 đến 200).
- Mỗi thước đo tính **riêng từng chuỗi** trên cửa sổ đánh giá, sau đó báo trung bình và trung vị qua các chuỗi (và theo nhóm mức bán khi cần). Lý do: quy mô chuỗi rất khác nhau, có một chuỗi bán rất lớn (`FOODS_3_090_CA_3`).
- Ngày 25/12 bị loại khỏi mọi thước đo (tham số `mask`); số ngày bị loại được ghi cùng kết quả.
- sMAPE: ngày thật = 0 và dự báo = 0 tính là 0 (dự báo đúng). [Giả định], đây là quy ước tự chọn. Ngày thật = 0 mà dự báo > 0 luôn nhận điểm xấu nhất (200%).
- Hạn chế:
  - sMAPE và MAE có xu hướng ưa dự báo gần 0 với chuỗi nhiều ngày bán 0. Đã kiểm chứng trên validation (mục 11): ở nhóm bán thấp và vừa, dự báo toàn số 0 có MAE trung vị thấp hơn cả naive và seasonal naive; ở nhóm bán thấp sMAPE cũng vậy. Vì vậy sMAPE chỉ là thước phụ, không dùng một mình để chọn mô hình.
  - Mẫu số của RMSSE có chứa các ngày 25/12 trong lịch sử (doanh số 0 rồi tăng lại), làm nó hơi lớn hơn thực tế. [Giả định] ảnh hưởng nhỏ vì chỉ 4 ngày trong hơn 1.600 ngày; chưa kiểm chứng.

## 11. Baseline 1: naive và seasonal naive (Tuần 2)

Code ở `src/forecast/baselines.py` và `src/evaluation/backtest.py`; chạy bằng `python -m src.evaluation.backtest --split valid`. Test ở `tests/test_baselines.py`.

Cách chạy:
- Naive: mọi ngày tương lai bằng số bán của ngày cuối cùng đã biết. Seasonal naive (chu kỳ 7 ngày): lặp lại 7 ngày cuối. Thêm một dòng tham chiếu "toàn số 0" (không phải baseline trong kế hoạch).
- Dự báo lùi theo cửa sổ (rolling-origin): mốc dự báo là cuối ngày ngay trước mỗi cửa sổ 28 ngày; chỉ dùng số bán thật đến mốc đó. 6 cửa sổ cho validation. Mỗi thước đo tính cho từng chuỗi trên từng cửa sổ, rồi lấy trung bình các cửa sổ, rồi báo trung bình và trung vị qua 297 chuỗi.
- [Giả định] Dự báo mỗi 28 ngày một lần. Mô phỏng kho (Tuần 5–9) có thể cần cập nhật dự báo thường xuyên hơn; khi đó mở rộng.
- Validation không có ngày 25/12 (0 ngày bị loại). Với các mốc dự báo hiện tại, 7 ngày cuối của lịch sử không chứa 25/12 nên seasonal naive không bị ảnh hưởng; trung bình trượt (mục 5) thì có thể.
- Chưa chạy trên test. Các baseline có tham số (trung bình trượt) được chọn trên validation; test chạy một lần ở mục 7.

Kết quả trên validation (trung bình / trung vị qua 297 chuỗi):

| Cách | MAE | RMSE | sMAPE (%) | RMSSE |
|---|---|---|---|---|
| Toàn số 0 (tham chiếu) | 2,268 / 1,024 | 2,937 / 1,567 | 100,585 / 100,000 | 0,880 / 0,836 |
| Naive | 1,980 / 1,226 | 2,486 / 1,601 | 85,504 / 92,653 | 0,892 / 0,831 |
| Seasonal naive (7) | 1,770 / 1,179 | 2,426 / 1,672 | 83,920 / 92,191 | 0,882 / 0,876 |

MAE trung vị theo nhóm mức bán (0 = thấp, 2 = cao; nhóm là tương đối trong từng cửa hàng):

| Cách | Nhóm 0 | Nhóm 1 | Nhóm 2 |
|---|---|---|---|
| Toàn số 0 | 0,423 | 1,042 | 2,601 |
| Naive | 0,613 | 1,250 | 2,512 |
| Seasonal naive (7) | 0,619 | 1,214 | 2,393 |

Quan sát [Đã kiểm chứng trên validation]:
- Ở nhóm bán thấp và vừa, "toàn số 0" có MAE trung vị thấp hơn naive và seasonal naive; ở nhóm bán cao thì ngược lại. Mô hình sau này phải thắng cả "toàn số 0" ở các chuỗi thưa.
- sMAPE của "toàn số 0" bằng 200 × tỉ lệ ngày có bán; nhóm thấp khoảng 29% và nhóm cao khoảng 70% ngày có bán, cùng khoảng với quan sát ở Tuần 1.
- Seasonal naive chỉ dựa trên một tuần nên bị nhiễu ở chuỗi thưa: kết quả ở mục 12 phù hợp với giả định này (RMSSE trung bình giảm dần khi lấy trung bình theo thứ trên nhiều tuần hơn).
- Trung bình lớn hơn trung vị nhiều: một số chuỗi bán nhiều kéo trung bình lên.

Giải thích [Giả định, chưa kiểm chứng]:
- Seasonal naive chỉ dựa trên một tuần nên bị nhiễu ở chuỗi thưa; sẽ kiểm tra bằng trung bình trượt nhiều tuần (Mục 5).
- RMSSE nhỏ hơn 1 ở cả ba cách vì mẫu số (sai số của việc đoán "mai bằng hôm nay") lớn khi chuỗi nhiễu, nên RMSSE dưới 1 chưa có nghĩa là dự báo tốt.
- Chưa có cách đo độ chắc chắn của các chênh lệch nhỏ (297 chuỗi, 6 cửa sổ), nên không xếp hạng các cách có chênh lệch nhỏ.

## 12. Baseline 2: trung bình trượt (Tuần 2)

Code: `src/forecast/baselines.py` (`moving_average_forecast`, `seasonal_moving_average_forecast`) và `src/evaluation/run_baselines.py`. Test: `tests/test_moving_average.py`. Cấu hình đã chọn: `configs/baselines.toml`.

Cách làm:
- Trung bình trượt: mọi ngày trong 28 ngày tới bằng trung bình `k` ngày gần nhất; thử k = 7, 14, 28, 56, 112 (đều là bội của 7).
- Trung bình theo thứ (ngoài kế hoạch, thêm để kiểm tra giả định ở mục 11): mỗi thứ bằng trung bình của cùng thứ đó trong w tuần gần nhất; thử w = 2, 4, 8, 16 (w = 1 chính là seasonal naive).
- Ngày đóng cửa (25/12) bị bỏ khỏi phép trung bình. Validation không có ngày 25/12 nên chưa ảnh hưởng số liệu; mới kiểm tra bằng test với dữ liệu giả, và sẽ có tác dụng ở test (mốc d_1801 có 25/12 trong 28 ngày gần nhất).
- Quy tắc chọn (đặt trước khi chạy): trong mỗi nhóm, chọn cấu hình có RMSSE trung bình thấp nhất trên validation. Kết quả: trung bình trượt 28 ngày; trung bình theo thứ 16 tuần.
- Số của cấu hình đã chọn hơi lạc quan vì được chọn trên chính validation; số sạch là số trên test (Tuần 2 mục 7).

Kết quả trên validation (trung bình / trung vị qua 297 chuỗi):

| Cách | RMSSE | MAE | RMSE | sMAPE (%) |
|---|---|---|---|---|
| Toàn số 0 (tham chiếu) | 0,880 / 0,836 | 2,268 / 1,024 | 2,937 / 1,567 | 100,585 / 100,000 |
| Naive | 0,892 / 0,831 | 1,980 / 1,226 | 2,486 / 1,601 | 85,504 / 92,653 |
| Seasonal naive (7) | 0,882 / 0,876 | 1,770 / 1,179 | 2,426 / 1,672 | 83,920 / 92,191 |
| Trung bình trượt 7 ngày | 0,691 / 0,681 | 1,502 / 0,991 | 1,939 / 1,293 | 103,620 / 103,649 |
| Trung bình trượt 14 ngày | 0,678 / 0,671 | 1,504 / 0,980 | 1,927 / 1,278 | 109,443 / 106,375 |
| **Trung bình trượt 28 ngày (chọn)** | 0,678 / 0,682 | 1,534 / 0,976 | 1,955 / 1,272 | 113,564 / 110,419 |
| Trung bình trượt 56 ngày | 0,687 / 0,686 | 1,553 / 0,982 | 1,975 / 1,274 | 119,001 / 117,478 |
| Trung bình trượt 112 ngày | 0,698 / 0,684 | 1,587 / 1,019 | 2,011 / 1,309 | 123,601 / 122,627 |
| Trung bình theo thứ 2 tuần | 0,786 / 0,780 | 1,652 / 1,089 | 2,194 / 1,477 | 91,298 / 100,024 |
| Trung bình theo thứ 4 tuần | 0,733 / 0,736 | 1,589 / 1,036 | 2,069 / 1,380 | 101,591 / 107,016 |
| Trung bình theo thứ 8 tuần | 0,714 / 0,708 | 1,564 / 1,012 | 2,018 / 1,322 | 112,010 / 113,818 |
| **Trung bình theo thứ 16 tuần (chọn)** | 0,709 / 0,694 | 1,573 / 1,041 | 2,014 / 1,326 | 119,955 / 122,907 |

Quan sát [Đã kiểm chứng trên validation]:
- Mọi cấu hình trung bình đều có RMSSE, MAE trung bình và RMSE trung bình thấp hơn "toàn số 0", naive và seasonal naive. Cấu hình tốt nhất (28 ngày) có RMSSE trung bình 0,678, thấp hơn khoảng 23–24% so với "toàn số 0" và naive.
- RMSSE trung bình của trung bình theo thứ giảm đều khi tăng số tuần: 0,882 (1 tuần = seasonal naive), 0,786, 0,733, 0,714, 0,709. Tốt nhất ở đầu lưới (16 tuần), mức giảm từ 8 lên 16 tuần chỉ 0,005; chưa thử dài hơn.
- Trung bình phẳng 28 ngày có RMSSE thấp hơn trung bình theo thứ 16 tuần (0,678 so với 0,709); ở cùng độ dài lịch sử 112 ngày là 0,698 và 0,709 (gần như hòa).
- 14 ngày và 28 ngày hòa về RMSSE trung bình ở ba chữ số thập phân; chọn 28 ngày theo đúng quy tắc.
- sMAPE xếp hạng ngược với RMSSE, MAE, RMSE: trung bình trượt 28 ngày có sMAPE 113,6, cao hơn "toàn số 0" (100,6) và naive (85,5), vì dự báo trung bình hầu như không bao giờ đúng bằng 0 nên mọi ngày bán 0 bị điểm 200%. Xác nhận sMAPE không dùng để chọn mô hình.
- Theo nhóm mức bán (trung bình trượt 28 ngày, MAE trung vị): nhóm 0: 0,583 (cao hơn "toàn số 0": 0,423); nhóm 1: 0,992 (thấp hơn 1,042); nhóm 2: 1,918 (thấp hơn 2,601). RMSE trung vị thấp hơn "toàn số 0" ở cả ba nhóm. RMSSE trung bình theo nhóm gần như đều: 0,680; 0,668; 0,685.

Giải thích [Giả định, chưa kiểm chứng]:
- Trung bình theo thứ không hơn trung bình phẳng vì ước lượng 7 mức riêng cho từng chuỗi thưa từ ít dữ liệu quá nhiễu. Mô hình chung học từ nhiều chuỗi (Tuần 3) có thể khai thác mùa vụ tuần tốt hơn.

Quyết định: baseline điểm dự báo mạnh nhất để so sánh ở các tuần sau là trung bình trượt 28 ngày.

Hạn chế:
- Chưa có cách đo độ chắc chắn cho các chênh lệch nhỏ (297 chuỗi, 6 cửa sổ).
- Dự báo là một con số cho mỗi ngày, không phải khoảng, nên chưa dùng trực tiếp cho đặt hàng; đó là việc của Tuần 4.
- Cách bỏ ngày đóng cửa khỏi phép trung bình chỉ mới kiểm tra bằng dữ liệu giả.

## 13. Baseline 3: Croston SBA và ETS (Tuần 2)

Code: `src/forecast/statistical.py` (`croston_sba_forecast`, `ets_forecast`) và `src/evaluation/run_statistical.py`. Test: `tests/test_statistical.py`. Thư viện: `statsforecast` 2.1.1.

Cách làm:
- ETS: thư viện tự chọn kiểu sai số, xu hướng và mùa vụ cho từng chuỗi (`"ZZZ"`); chu kỳ mùa vụ 7 ngày [Giả định]; khớp lại từ đầu trên từng chuỗi ở từng cửa sổ.
- Croston (biến thể SBA): dự báo một mức phẳng bằng lượng bán trung bình mỗi lần có bán chia cho số ngày trung bình giữa hai lần bán. Thêm làm tham chiếu cho chuỗi thưa (ngoài kế hoạch).
- ARIMA không làm: dự kiến chạy chậm hơn ETS nhiều (chưa đo) [Quyết định].
- Mỗi chuỗi dùng lịch sử từ ngày bán đầu tiên; chuỗi chưa bán lần nào trước mốc dự báo thì dự báo bằng 0 (không có trường hợp này trên validation). Dự báo âm được đặt về 0 [Giả định].
- Thư viện có phương án dự phòng (trung bình toàn lịch sử) khi ETS lỗi trên một chuỗi; không đếm được số lần dùng.
- Chạy trên validation, 6 cửa sổ, cùng cách tính như mục 11. Thời gian trên máy tác giả: Croston khoảng 12 giây, ETS khoảng 233 giây.

Kết quả trên validation (trung bình / trung vị qua 297 chuỗi):

| Cách | RMSSE | MAE | RMSE | sMAPE (%) |
|---|---|---|---|---|
| Toàn số 0 (tham chiếu) | 0,880 / 0,836 | 2,268 / 1,024 | 2,937 / 1,567 | 100,585 / 100,000 |
| Naive | 0,892 / 0,831 | 1,980 / 1,226 | 2,486 / 1,601 | 85,504 / 92,653 |
| Seasonal naive (7) | 0,882 / 0,876 | 1,770 / 1,179 | 2,426 / 1,672 | 83,920 / 92,191 |
| Trung bình trượt 28 ngày | 0,678 / 0,682 | 1,534 / 0,976 | 1,955 / 1,272 | 113,564 / 110,419 |
| Trung bình theo thứ 16 tuần | 0,709 / 0,694 | 1,573 / 1,041 | 2,014 / 1,326 | 119,955 / 122,907 |
| Croston SBA | 0,725 / 0,702 | 1,680 / 1,084 | 2,088 / 1,338 | 130,504 / 128,667 |
| **ETS (ZZZ, chu kỳ 7)** | 0,674 / 0,659 | 1,453 / 0,962 | 1,879 / 1,266 | 119,041 / 112,666 |

RMSSE trung bình theo nhóm mức bán (0 = thấp, 2 = cao; nhóm là tương đối trong từng cửa hàng):

| Cách | Nhóm 0 | Nhóm 1 | Nhóm 2 |
|---|---|---|---|
| Trung bình trượt 28 ngày | 0,680 | 0,668 | 0,685 |
| Croston SBA | 0,712 | 0,698 | 0,765 |
| ETS | 0,679 | 0,669 | 0,674 |

Quan sát [Đã kiểm chứng trên validation]:
- ETS có RMSSE trung bình thấp nhất (0,674), gần như hòa với trung bình trượt 28 ngày (0,678). ETS có MAE trung bình thấp hơn khoảng 5% (1,453 so với 1,534) và RMSSE thấp hơn ở 57,2% số chuỗi (khoảng 170 trên 297).
- Lợi thế của ETS tập trung ở nhóm bán nhiều: nhóm 2 có RMSSE 0,674 so với 0,685 và MAE trung bình 2,744 so với 2,985; nhóm 0 và 1 hai cách hòa.
- Croston SBA kém trung bình trượt 28 ngày (RMSSE 0,725 so với 0,678), thắng ở 48,8% số chuỗi, và cũng không tốt hơn ở nhóm bán thấp (0,712 so với 0,680).
- ETS dự báo âm (đặt về 0) ở 3.490 trên 49.896 dự báo (6,99%).
- sMAPE xếp hạng ngược với RMSSE, MAE, RMSE (ETS 119,0; Croston 130,5; naive 85,5), vì dự báo trung bình hầu như không bao giờ đúng bằng 0.
- Chạy thử trên 30 chuỗi (mỗi 10 chuỗi lấy một) cho thứ hạng khác lần chạy đủ (ví dụ Croston 0,674 so với 0,725); 30 chuỗi quá ít để xếp hạng.

Giải thích [Giả định, chưa kiểm chứng]:
- Lợi thế của ETS ở nhóm bán nhiều có thể do chuỗi đó có đủ dữ liệu để ước lượng mùa vụ tuần; chuỗi thưa thì không.
- Chưa có giải thích đã kiểm chứng cho việc Croston không giúp ở chuỗi thưa.

Quyết định: hai mốc điểm dự báo mạnh nhất để so sánh ở các tuần sau là ETS và trung bình trượt 28 ngày (gần như hòa về RMSSE). Test chạy một lần ở mục 7 cho tất cả cách ở bảng trên.

Hạn chế:
- Chênh lệch giữa ETS và trung bình trượt nhỏ; chưa có cách đo độ chắc chắn (297 chuỗi, 6 cửa sổ, các chuỗi cùng cửa hàng và cùng ngày nên không độc lập).
- MAE trung bình của nhóm 2 bị kéo bởi một chuỗi rất lớn (`FOODS_3_090_CA_3`); đã kiểm tra ở mục 14.
- Dự báo âm của ETS tập trung ở một số chuỗi (xem mục 14); chưa biết phương án dự phòng của thư viện được dùng bao nhiêu lần.
- Dự báo vẫn là một con số mỗi ngày, chưa phải khoảng (Tuần 4).

## 14. Kết quả baseline cuối cùng trên validation và test (Tuần 2)

Code: `src/evaluation/run_final_baselines.py`. Kết quả lưu ở `reports/baselines/`: `summary.csv` (điểm theo tập, cách, nhóm mức bán), `per_series_valid.csv` và `per_series_test.csv` (điểm từng chuỗi), `run_info.json` (phiên bản thư viện, mốc ngày, cấu hình). Chạy lại: `python -m src.evaluation.run_final_baselines`.

Cách làm:
- 7 cách: toàn số 0 (tham chiếu), naive, seasonal naive, trung bình trượt 28 ngày, trung bình theo thứ 16 tuần, Croston SBA, ETS. Cấu hình lấy từ `configs/baselines.toml` (chọn trên validation).
- Test chạy một lần, cho cả 7 cách, không chỉnh cấu hình theo kết quả test. Script từ chối chạy lại nếu file kết quả test đã tồn tại (cần `--overwrite`).
- Từ nay, mô hình ở các tuần sau được chọn trên validation; test chỉ dùng khi chốt kết quả.
- Ngày 25/12 bị loại khỏi thước đo (test có 1 ngày, validation 0 ngày) và khỏi các phép trung bình. ETS và Croston vẫn thấy số 0 của ngày 25/12 trong lịch sử [Giả định: ảnh hưởng nhỏ, chưa kiểm chứng].

Kết quả trên test (trung bình / trung vị qua 297 chuỗi):

| Cách | RMSSE | MAE | RMSE | sMAPE (%) |
|---|---|---|---|---|
| Toàn số 0 (tham chiếu) | 0,963 / 0,924 | 2,378 / 1,131 | 3,072 / 1,731 | 109,726 / 112,522 |
| Naive | 0,954 / 0,916 | 2,086 / 1,384 | 2,621 / 1,784 | 89,521 / 96,516 |
| Seasonal naive (7) | 0,944 / 0,931 | 1,890 / 1,306 | 2,554 / 1,810 | 89,708 / 95,768 |
| Trung bình trượt 28 ngày | 0,728 / 0,698 | 1,644 / 1,059 | 2,082 / 1,366 | 112,626 / 111,467 |
| Trung bình theo thứ 16 tuần | 0,754 / 0,714 | 1,664 / 1,087 | 2,118 / 1,438 | 119,984 / 118,437 |
| Croston SBA | 0,761 / 0,723 | 1,734 / 1,111 | 2,166 / 1,413 | 124,709 / 121,738 |
| **ETS (ZZZ, chu kỳ 7)** | 0,721 / 0,691 | 1,557 / 1,047 | 1,997 / 1,369 | 117,872 / 113,745 |

RMSSE trung bình, validation so với test:

| Cách | Validation | Test |
|---|---|---|
| Toàn số 0 | 0,880 | 0,963 |
| Naive | 0,892 | 0,954 |
| Seasonal naive (7) | 0,882 | 0,944 |
| Trung bình trượt 28 ngày | 0,678 | 0,728 |
| Trung bình theo thứ 16 tuần | 0,709 | 0,754 |
| Croston SBA | 0,725 | 0,761 |
| ETS | 0,674 | 0,721 |

Quan sát [Đã kiểm chứng]:
- Thứ hạng bốn cách đầu theo RMSSE trung bình giống nhau ở validation và test: ETS, trung bình trượt 28 ngày, trung bình theo thứ 16 tuần, Croston. Ba cách còn lại kém hẳn (RMSSE 0,944 đến 0,963 trên test). Chạy lại validation cho đúng số của lần chạy trước.
- Mọi cách đều tệ hơn trên test: RMSSE trung bình tăng 0,036 đến 0,083, MAE trung bình tăng khoảng 0,05 đến 0,12. Các cách không chỉnh gì (toàn số 0, naive, seasonal naive) tệ đi bằng hoặc nhiều hơn các cách đã chọn cấu hình, nên phần lớn chênh lệch không đến từ việc chọn cấu hình trên validation.
- ETS so với trung bình trượt 28 ngày: RMSSE trung bình chênh 0,007 trên test (0,004 trên validation); MAE trung bình thấp hơn 5,3% ở cả hai tập.
- Chuỗi `FOODS_3_090_CA_3` bán trung bình 117,8 đơn vị/ngày (validation) và 89,5 (test). MAE của chuỗi này: toàn số 0 là 117,756 / 89,493; trung bình trượt 28 ngày là 46,381 / 50,471; ETS là 33,592 / 37,615 (validation / test). Chuỗi này đóng góp 0,390 vào MAE trung bình 2,268 của "toàn số 0" (validation). Bỏ nó ra thì lợi thế MAE trung bình của ETS so với trung bình trượt còn khoảng một nửa: 0,081 còn 0,039 (validation), 0,087 còn 0,043 (test), tức khoảng 2,8–2,9% thay vì 5,3%. RMSSE không bị ảnh hưởng theo cách này.
- Dự báo âm của ETS (đặt về 0): 72 trên 297 chuỗi (validation) và 62 trên 297 chuỗi (test) có ít nhất một dự báo âm. Tỉ lệ dự báo âm trung bình theo nhóm mức bán 0 / 1 / 2: 5,4% / 6,1% / 9,5% (validation), 4,5% / 3,9% / 5,9% (test). Có chuỗi bị âm ở 120 trên 168 ngày (validation), tối đa 144 trên 168 ngày (test).
- sMAPE xếp hạng ngược với RMSSE, MAE, RMSE (xem mục 11–13).

Giải thích [Giả định, chưa kiểm chứng]:
- Giai đoạn test (12/2015 đến 5/2016) khó dự báo hơn validation; chưa tìm nguyên nhân.
- 120 = 5 × 24 và 144 = 6 × 24 (mỗi thứ trong tuần xuất hiện 24 lần trong 168 ngày) gợi ý ETS đẩy một số thứ trong tuần xuống dưới 0 (mô hình mùa vụ dạng cộng).

Quyết định: hai mốc điểm dự báo mạnh nhất để so sánh ở các tuần sau là ETS và trung bình trượt 28 ngày; báo kết quả test lấy từ `reports/baselines/summary.csv`. Khi so sánh, nhìn RMSSE trước (không bị một chuỗi lớn kéo).

Hạn chế:
- Chênh lệch giữa ETS và trung bình trượt nhỏ; chưa có khoảng tin cậy. Các chuỗi cùng cửa hàng và cùng ngày nên không độc lập.
- Test chỉ có 6 cửa sổ 28 ngày trong một giai đoạn; mỗi lần dự báo 28 ngày một lần (mô phỏng kho có thể cần cập nhật dự báo thường xuyên hơn).
- Dự báo vẫn là một con số mỗi ngày, chưa phải khoảng (Tuần 4).
- Thời gian chạy ETS trên máy tác giả dao động: khoảng 233 giây (lần chạy riêng), 346 giây (validation) và 499 giây (test) trong lần chạy cuối.