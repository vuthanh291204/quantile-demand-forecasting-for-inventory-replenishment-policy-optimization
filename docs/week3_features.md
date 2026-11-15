# Tuần 3 — Mục 1–2: đặc trưng và kiểm tra rò rỉ

Đã làm bộ tạo đặc trưng và các kiểm tra rò rỉ ở cấp đặc trưng. Chưa huấn luyện
LightGBM, chưa thiết lập MLflow tracking và chưa có điểm dự báo mới.

## Dự án đang ở đâu?

- Tập con đã cố định: 297 chuỗi FOODS, ba cửa hàng CA_3, TX_2, WI_1;
  danh sách ở `configs/subset_ids.csv`.
- Notebook `01_explore_tables.ipynb` chọn tập con; notebook
  `02_explore_tables.ipynb` nối doanh số, lịch, giá và khám phá dữ liệu.
- `data/processed/subset_long.parquet`: 576.477 dòng = 297 × 1.941 ngày,
  có cả ngày trước khi sản phẩm bắt đầu bán. Đây là đầu vào của bộ đặc trưng.
- `configs/split.toml`: train kết thúc d_1605, validation kết thúc d_1773,
  test kết thúc d_1941. Mỗi cửa sổ dự báo dài 28 ngày.
- Tuần 2 đã có bảy cách dự báo, thước đo dùng lại được và kết quả lưu trong
  `reports/baselines/`. Hai mốc chính để so sánh là ETS và trung bình trượt
  28 ngày; chọn cấu hình mô hình mới bằng validation.
- LightGBM và MLflow đã có trong `.venv` khi kiểm tra ngày 14/11/2026.
  Có thư viện chưa có nghĩa là đã có mô hình hoặc thí nghiệm được ghi lại.

## Đặc trưng nghĩa là gì?

Đặc trưng là thông tin đầu vào giúp mô hình đoán số bán. Nhãn `sales` là
đáp án cần học; tuyệt đối không nằm trong danh sách đầu vào.

Các lựa chọn ban đầu lưu ở `configs/features.toml`. Độ dài lag và cửa sổ
là **lựa chọn thí nghiệm**, chưa phải cấu hình tối ưu đã kiểm chứng.

- `sales_lag_7/14/28`: số bán của chính chuỗi đó cách ngày cần dự báo
  7, 14 hoặc 28 ngày. Ví dụ minh họa: dự báo thứ Hai, lag 7 là số bán thứ Hai
  tuần trước, không phải tổng số bán tuần trước.
- `sales_roll_mean_7/28`: số bán trung bình trong 7 hoặc 28 ngày lịch
  ngay trước ngày cần dự báo. Phải dịch một ngày trước khi tính trung bình
  (`shift(1)` rồi `rolling`), để đáp án hôm nay không lọt vào đầu vào.
  Bỏ ngày 25/12 khỏi mẫu số và tổng số bán khi tính trung bình,
  theo giả định đóng cửa đã dùng ở tuần 2.
- `day_of_week`, `month`, `day_of_month`, `year`: vị trí của ngày cần dự báo
  trong lịch. `day_of_week` dùng thứ Hai = 0, Chủ Nhật = 6; khác cột `wday`
  gốc của M5, vốn bắt đầu từ thứ Bảy.
- `event_name_1/2`, `event_type_1/2`, `has_event`: tên, loại và cờ sự kiện
  của ngày cần dự báo. Ngày không có sự kiện dùng chuỗi `none`.
- `snap`: chọn đúng cờ SNAP của bang tương ứng với cửa hàng. Không lấy cờ
  của một bang khác làm đại diện cho mọi cửa hàng.
- `is_closed`: cờ 25/12 theo `configs/split.toml`.
- `sell_price_lag_1`: giá ngày trước đó; ở bước đầu là giá tại mốc dự báo.
  Trong những ngày tương lai chưa quan sát, giữ giá cuối cùng đã biết tại mốc.
- `item_id`, `dept_id`, `store_id`, `state_id`: thông tin sản phẩm và cửa hàng,
  để chuẩn bị học một mô hình chung trên nhiều chuỗi.

Có tổng cộng 21 cột đầu vào. `id`, `d`, `day`, `date` dùng để đối chiếu và
chia thời gian; `sales` là nhãn; `features_ready` là cờ kiểm tra đủ lịch sử.
Những cột phụ này không nằm trong `FeatureConfig.feature_columns`.
Khi huấn luyện phải lấy đúng danh sách này, đồng thời học danh sách giá trị
phân loại từ tập train rồi áp dụng lại cho validation/test.

## Vì sao cần xử lý riêng cửa sổ 28 ngày?

Ví dụ minh họa: đứng ở cuối d_1605 để dự báo d_1606–d_1633. Với d_1613,
lag 7 là số bán d_1606. Nhưng tại mốc d_1605 ta chưa biết số bán thật d_1606.
Nếu lấy trực tiếp từ bảng validation, kết quả đánh giá sẽ lạc quan sai.

**Quyết định ban đầu:** chuẩn bị cho cách dự báo đệ quy, nghĩa là dự báo
lần lượt từng ngày và dùng dự báo của ngày trước làm lịch sử cho ngày sau.
Đây là lựa chọn kỹ thuật, chưa có bằng chứng tốt hơn các cách khác.

1. `prepare_forecast_frame(long, origin_day, horizon)` giữ lịch sử đến mốc,
   xóa mọi `sales` sau mốc và thay giá tương lai bằng giá cuối cùng đã biết.
2. `build_features(frame, cfg)` tạo đầu vào. Ban đầu chỉ ngày đầu tiên của
   cửa sổ có `features_ready=True`.
3. Ở mục huấn luyện, mô hình sẽ dự báo ngày đó và điền **dự báo** vào `sales`
   của khung đã xóa đáp án, rồi tạo đầu vào ngày kế tiếp. Không điền số bán thật.
4. Chỉ khi chuyển sang cửa sổ 28 ngày tiếp theo mới nhận thêm số bán thật
   của cửa sổ trước, giống baseline tuần 2.

Hàm tạo đặc trưng chưa phải vòng lặp dự báo hoàn chỉnh. Bước 3 và đánh giá
LightGBM đủ sáu cửa sổ sẽ làm ở các đầu mục sau. Sai số có thể tích lũy khi
dùng dự báo làm đầu vào; cần đo thay vì mặc định mô hình sẽ tốt hơn baseline.

## Giả định và cách xử lý dữ liệu

- **Giả định:** lịch sự kiện và lịch SNAP của 28 ngày tới đã biết tại mốc.
  M5 có các cột này, nhưng không có lịch sử thời điểm công bố lịch.
- Không giả định biết giá thật của cả 28 ngày tới. Dùng giá lịch sử và giữ
  mức cuối cùng đã biết; mức giữ nguyên này có thể khác giá thực tế.
- Giữ giá 0,33 đã phát hiện ở tuần 1: chưa có bằng chứng đây là lỗi.
  Chưa thêm tỉ lệ giá so với trung vị toàn chuỗi, vì phép tính trên toàn bộ
  thời gian sẽ thấy giá tương lai.
- Chỉ dùng một dòng train sau khi có đủ 28 ngày lịch sử quan sát kể từ lần
  bán dương đầu tiên. Không coi ngày trước khi sản phẩm bắt đầu bán là
  nhu cầu bằng 0. Đây là quy tắc đủ lịch sử được chọn cho bộ đặc trưng ban đầu.
- Giữ nhãn số bán 25/12 trong train và thêm cờ đóng cửa; loại ngày này khỏi
  phép trung bình trượt. Quy tắc loại ngày khỏi thước đo tuần 2 vẫn áp dụng
  khi đánh giá mô hình ở các bước sau.
- Các cột EDA như `live`, `first_date`, `volume_group`, `mean_live`, `rel`
  không làm đầu vào. Bộ tạo đặc trưng dùng danh sách cột cho phép rõ ràng.
- Hạn chế có sẵn: notebook chọn tập con và chia mức bán bằng thống kê trên
  toàn bộ d_1–d_1941, có cả giai đoạn validation/test. Giữ nguyên tập con
  để so sánh với baseline đã lưu, nhưng kết quả là trên tập con đã chọn này;
  chưa thể khẳng định quá trình chọn tập hoàn toàn chỉ nhìn quá khứ.
  Nếu cần thí nghiệm chọn mẫu chỉ dùng train, phải thiết kế riêng và chạy lại
  cả baseline. Không dùng mức bán này để huấn luyện mô hình.

## Chạy và kiểm tra

Từ thư mục gốc repo, trong PowerShell:

```powershell
.\.venv\Scripts\python.exe -m src.data.features
.\.venv\Scripts\python.exe -m pytest tests\test_features.py -q -p no:cacheprovider
```

Lệnh đầu chỉ in tóm tắt, không lưu file. Nếu muốn lưu bảng train để xem:

```powershell
.\.venv\Scripts\python.exe -m src.data.features --output data/processed/train_features.parquet
```

File parquet này được `.gitignore` bỏ qua như các dữ liệu xử lý khác.
Kết quả đã chạy trên dữ liệu thật:

- Train đến d_1605: **391.610 dòng**, **297 chuỗi**, **21 đặc trưng**.
- Số giá trị thiếu trong các đặc trưng của bảng train: **0**.
- Số doanh số tương lai còn giữ trong đầu vào cửa sổ validation đầu: **0**.
- Trước vòng lặp dự báo: **297 dòng** tương lai đủ đầu vào, đều là **d_1606**.
  Các ngày sau cần điền dự báo trước khi đủ đầu vào.

Tests dùng dữ liệu giả có đáp án tính tay, không dùng các con số giả làm
kết quả thực nghiệm. Đã kiểm tra lag và trung bình không lẫn chuỗi,
không dùng nhãn hiện tại, cắt train đúng mốc, xử lý trước lần bán đầu,
ngày đóng cửa, SNAP, xóa doanh số/giá tương lai, dùng dự báo cho lag ở
bước sau và từ chối ngày trùng hoặc thiếu. Đây là kiểm tra ở cấp đặc trưng;
vẫn cần kiểm tra vòng lặp dự báo sau khi có mô hình.

## Mục 2: kiểm tra rò rỉ trên sáu cửa sổ validation

Rò rỉ nghĩa là đầu vào hoặc quá trình huấn luyện đã thấy thông tin mà tại
mốc dự báo chưa thể biết. Ví dụ: tính trung bình trượt có cả số bán thật
của ngày đang cần dự báo, hoặc dùng lag thật nằm trong cửa sổ tương lai.

Đã thêm `src/evaluation/check_leakage.py`. Chạy khi `.venv` đang được kích hoạt:

```powershell
python -m src.evaluation.check_leakage
```

Script kiểm tra những điều sau, và dừng bằng lỗi nếu bất kỳ kiểm tra nào sai:

1. Thay toàn bộ doanh số và giá sau d_1605 bằng giá trị khác. Bảng train,
   cả đầu vào và nhãn, phải giữ nguyên. Danh sách đầu vào không được chứa
   nhãn, cột phụ hoặc thống kê EDA toàn giai đoạn.
2. Ở mỗi cửa sổ validation, thay doanh số và giá sau **mốc riêng của cửa sổ**.
   Khung đầu vào sau khi xóa đáp án phải giữ nguyên. Trước khi điền dự báo,
   chỉ ngày đầu tiên đủ đầu vào và không còn số bán thật nào sau mốc.
3. Giá ở mọi ngày tương lai phải bằng giá cuối cùng đã biết tại mốc,
   không phải giá tương lai trong dữ liệu gốc.
4. Điền một dãy số giả, khác nhau theo chuỗi và ngày, vào vị trí dành cho dự báo.
   Với cả 28 ngày, đối chiếu từng lag và trung bình trượt với phép tính NumPy
   độc lập chỉ lấy các ngày trước ngày đích. Cách này kiểm tra số điền cho
   hôm nay hoặc ngày sau không lọt vào đầu vào hôm nay, và không lẫn chuỗi.

Dãy số giả ở bước 4 chỉ là dụng cụ kiểm tra, **không phải kết quả dự báo**.
Script không huấn luyện mô hình, không tính điểm test và không chọn tham số.
Trong cửa sổ sau, số bán thật của cửa sổ trước đã xảy ra nên được dùng làm
lịch sử; đây là quy tắc đánh giá theo cửa sổ đã dùng ở tuần 2.

Kết quả chi tiết lưu ở `reports/leakage_checks.json`. Có sáu cửa sổ,
mỗi cửa sổ kiểm tra 8.316 dòng giả = 297 chuỗi × 28 ngày,
tổng cộng 49.896 dòng. Nếu chạy thành công, dòng cuối phần kiểm tra là:

```text
PASS all 6 validation windows; no model fitted or test scored
```

Tests còn cố tình tạo bốn lỗi để kiểm tra bộ kiểm tra có phát hiện được:
giữ doanh số tương lai, dùng nhãn hôm nay trong trung bình, lấy giá tương lai
và đưa thống kê doanh số toàn giai đoạn vào train.

**Phạm vi kết luận:** bộ đặc trưng và cách chuẩn bị cửa sổ vượt qua các ca
kiểm tra này, với giả định lịch sự kiện/SNAP biết trước. Hạn chế chọn mẫu
bằng toàn bộ giai đoạn vẫn còn như đã ghi ở trên. Khi có mô hình vẫn phải
kiểm tra mã hóa phân loại chỉ học từ train, thời gian dữ liệu dùng để fit
và vòng lặp dự báo thực tế. Bước 4.1 đã có mã hóa phân loại chỉ lấy danh sách
nhóm từ train; việc áp dụng danh sách đó và vòng lặp đã qua test ở bước 4.3.

## Tiến độ theo đúng đầu mục trong file kế hoạch

- [x] 1. Tạo đặc trưng.
- [x] 2. Kiểm tra rò rỉ ở bộ đặc trưng và cách chuẩn bị sáu cửa sổ validation.
- [x] 3. Thiết lập MLflow để lưu tham số, kết quả và các file thí nghiệm.
- [x] 4. Huấn luyện LightGBM, triển khai dừng sớm theo mốc và chọn cấu hình
  trong ngân sách 500 cây; lần chạy thật chạm giới hạn, chưa hết cải thiện.
- [ ] 5. So sánh trên cùng sáu cửa sổ validation với baseline tuần 2.
- [ ] 6. Xem tầm quan trọng đặc trưng và ghi nhận xét có giới hạn.
- [ ] 7. Hoàn thiện đoạn báo cáo về đặc trưng, mô hình và kết quả thực đo.

Mục 1–2 đã được người thực hiện chạy và xác nhận, lưu trong commit `0159df7`.
Hướng dẫn mục 3 ở [week3_mlflow.md](week3_mlflow.md); đã chuẩn bị code ghi
thí nghiệm, người thực hiện đã chạy thành công và xác nhận thấy run trong giao diện.
Mục 3 lưu trong commit `44e6fd6`. Tiến độ mục 4 ở
[week3_lightgbm.md](week3_lightgbm.md): bước 4.1–4.3 đã xác nhận; mô hình
100 cây đã dự báo đủ 28 ngày của cửa sổ validation đầu, có test vòng lặp và
kết quả trong MLflow. Bước 4.4A đã chấm đủ 297 chuỗi trên cửa sổ đầu, RMSSE
trung bình 0,714618. Bước 4.4B chọn 500 cây trong phạm vi đã kiểm tra, RMSSE
0,696867 trên cửa sổ đầu; dừng do `max_rounds`. Mục 4 hoàn thành với giới hạn
chưa chứng minh hội tụ; chuyển sang so sánh sáu cửa sổ ở mục 5.
Chưa đánh giá LightGBM trên test.
