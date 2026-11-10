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
- Ngày SNAP bán cao hơn khoảng 13% (1,086 so với 0,958), đúng chiều ở cả 3 cửa hàng. Đây là mối liên hệ, chưa chứng minh nguyên nhân.
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

- Thước đo chính, không phụ thuộc quy mô: **RMSSE** (root mean squared scaled error) tính riêng từng chuỗi. Đây là thành phần từng chuỗi của WRMSSE, thước đo chính của cuộc thi M5 (nguồn: trang Evaluation của cuộc thi M5 trên Kaggle). Cách tính: sai số bình phương trung bình của dự báo trên cửa sổ đánh giá, chia cho trung bình bình phương mức thay đổi từ ngày này sang ngày kế tiếp trong lịch sử của chính chuỗi, rồi lấy căn. Phần lịch sử bắt đầu từ ngày bán đầu tiên của chuỗi.
- **Không dùng WRMSSE đầy đủ.** WRMSSE gộp nhiều mức tổng hợp của toàn bộ 30.490 chuỗi và có trọng số theo doanh thu; tập con 297 chuỗi và bài toán đặt hàng theo từng sản phẩm × cửa hàng không khớp với cách đó. Hệ quả: điểm RMSSE của đồ án **không so được** với bảng xếp hạng M5. Chưa dùng trọng số doanh thu. [Quyết định, xem lại nếu cần]
- Thước đo đọc theo đơn vị: **MAE**, **RMSE** (số lượng bán mỗi ngày). Thước phụ: **sMAPE** (phần trăm, 0 đến 200).
- Mỗi thước đo tính **riêng từng chuỗi** trên cửa sổ đánh giá, sau đó báo trung bình và trung vị qua các chuỗi (và theo nhóm mức bán khi cần). Lý do: quy mô chuỗi rất khác nhau, có một chuỗi bán rất lớn (`FOODS_3_090_CA_3`).
- Ngày 25/12 bị loại khỏi mọi thước đo (tham số `mask`); số ngày bị loại được ghi cùng kết quả.
- sMAPE: ngày thật = 0 và dự báo = 0 tính là 0 (dự báo đúng). [Giả định], đây là quy ước tự chọn. Ngày thật = 0 mà dự báo > 0 luôn nhận điểm xấu nhất (200%).
- Hạn chế:
  - sMAPE và MAE đều có xu hướng ưa dự báo gần 0 với chuỗi nhiều ngày bán 0 (ví dụ minh họa tính tay, chưa kiểm chứng trên dữ liệu). RMSE và RMSSE ưa dự báo gần trung bình hơn, nên các thước đo có thể xếp hạng khác nhau. Vì vậy sMAPE chỉ là thước phụ, không dùng một mình để chọn mô hình.
  - Mẫu số của RMSSE có chứa các ngày 25/12 trong lịch sử (doanh số 0 rồi tăng lại), làm nó hơi lớn hơn thực tế. [Giả định] ảnh hưởng nhỏ vì chỉ 4 ngày trong hơn 1.600 ngày; chưa kiểm chứng.