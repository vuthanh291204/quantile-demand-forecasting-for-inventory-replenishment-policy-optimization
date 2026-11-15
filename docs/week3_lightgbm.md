# Tuần 3 — Mục 4: huấn luyện LightGBM

## Cách thực hiện

Người thực hiện tự viết code và chạy các lệnh theo hướng dẫn trong hội thoại,
gửi output trước khi chuyển bước. Trợ lý cập nhật các file Markdown để ghi
nhận tiến độ, quyết định và kết quả đã có; chỉ làm thay phần code khi được yêu cầu.

Mốc trước: mục 3 đã hoàn thành và commit trong `44e6fd6` —
`Set up local MLflow experiment tracking`.

## 4.1. Chuẩn bị dữ liệu — đã xác nhận

Code do người thực hiện tạo: `src/forecast/lightgbm_model.py`.
Đã chạy `python -m src.forecast.lightgbm_model`, nhận kết quả:

```text
X shape: (391610, 21)
y shape: (391610,)
Categorical columns: ['item_id', 'dept_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
Object columns: []
Missing values: 0
Sales in X: False
```

Ý nghĩa:

- `X` là bảng đầu vào, gồm đúng 21 cột từ `FeatureConfig.feature_columns`.
- `y` là nhãn số bán thực tế, kiểu `float32`; không nằm trong `X`.
- Tám cột tên nhóm được chuyển thành `category`, để LightGBM xử lý như đặc
  trưng phân loại. Danh sách giá trị từng cột chỉ lấy từ train và giữ lại
  trong `category_levels` để dùng khi dự báo.
- Bảng train được tạo với mốc d_1605 qua `build_training_table`.
- Chưa có phép đo độ chính xác hoặc mô hình LightGBM đã huấn luyện ở bước này.

## 4.2. Huấn luyện kiểm tra lần đầu — đã xác nhận

Người thực hiện đã tạo `configs/lightgbm.toml`, bổ sung phần đọc cấu hình,
khởi tạo `LGBMRegressor`, gọi `fit(X, y)` và ghi một run MLflow tên
`lightgbm-initial-fit`. Code do người thực hiện viết và chạy theo hướng dẫn.

Cấu hình đã chạy: `objective="regression"`, `n_estimators=100`,
`learning_rate=0.05`, `num_leaves=31`, `min_child_samples=50`,
`random_state=29`, `n_jobs=4`, `verbosity=-1`.

Đây là **lựa chọn thí nghiệm ban đầu**, chưa được chọn theo độ chính xác
validation, không phải tham số tối ưu hoặc giả định chi phí vận hành.
Tối đa 100 cây dùng cho lần kiểm tra đầu;
số vòng huấn luyện của mô hình cuối sẽ được chọn bằng validation ở bước sau.

Run kiểm tra ghi tham số, số dòng/số đặc trưng, số vòng học thực tế,
file mô hình LightGBM dạng text, danh sách nhóm, tên cột và cấu hình/mã nguồn.
Các số đếm này không phải MAE/RMSSE và chưa có cơ sở kết luận LightGBM tốt
hơn baseline. Output người thực hiện gửi:

```text
Training started...
Run ID: 97efae350d604f16b0b7418ae7cc27b4
Status: FINISHED
Feature count: 21
Fitted iterations: 100
```

Người thực hiện đã xác nhận thấy run trong giao diện MLflow.

`objective="regression"` học bằng lỗi bình phương trên từng dòng; mục tiêu này
không đồng nhất với cách lấy trung bình RMSSE qua từng chuỗi ở tuần 2.
Ảnh hưởng của chuỗi bán nhiều cần xem khi đánh giá, thay vì mặc định mô hình
đã tối ưu đúng thước đo nghiên cứu.

Tham khảo: [LGBMRegressor](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)
và [tham số LightGBM](https://lightgbm.readthedocs.io/en/stable/Parameters.html).

## 4.3. Dự báo 28 ngày — đã xác nhận

Người thực hiện đã thêm hàm dự báo đệ quy vào
`src/forecast/lightgbm_model.py`, cùng test trong `tests/test_features.py`.
Code được người thực hiện viết và chạy theo hướng dẫn; trợ lý cập nhật tài liệu.

Luồng đã triển khai:

1. `prepare_forecast_frame` xóa số bán sau mốc và giữ giá tương lai ở mức
   cuối cùng đã biết. Lịch sự kiện/SNAP tiếp tục được giả định biết trước.
2. Mỗi ngày tạo đặc trưng bằng `build_features`, kiểm tra đủ mỗi chuỗi một dòng
   và đủ lịch sử; chọn đúng danh sách cột đã dùng khi huấn luyện.
3. Áp dụng lại `category_levels` lấy từ train. Nhóm mới ngoài danh sách được
   chuyển thành giá trị thiếu để LightGBM xử lý; không học thêm nhóm từ tương lai.
4. Dự báo, kiểm tra kết quả hữu hạn và đặt dự báo âm về 0. Ghi dự báo vào
   khung đầu vào theo đúng id để tạo đặc trưng cho ngày kế tiếp.
5. Với mô hình 100 cây, kiểm tra cửa sổ validation đầu tiên: d_1606–d_1633,
   297 chuỗi × 28 ngày = 8.316 dòng dự báo. Kết quả chạy đã khớp kích thước này.

[Quyết định ban đầu] Đặt dự báo âm về 0 giống quy tắc baseline tuần 2;
giữ giá trị số thực, chưa làm tròn thành số nguyên.

Test ghi lại đầu vào của mô hình giả qua 28 ngày, xác nhận ngày thứ 8 dùng
dự báo ngày đầu làm lag 7. Khi thay toàn bộ doanh số/giá thật sau mốc, cả đầu
vào ghi lại và dự báo phải giữ nguyên; dữ liệu gốc cũng không được thay đổi.

Lần chạy chính đã fit lại 100 cây theo cùng cấu hình và tạo run MLflow
`lightgbm-first-window-check`, lưu bảng dự báo vào artifacts. Output người
thực hiện gửi và xác nhận đã thấy trên giao diện:

```text
Forecast shape: (8316, 4)
Forecast series: 297
Forecast days: 1606 1633
Run ID: a9ab59523715493da9a8415eb182ed5c
Status: FINISHED
Feature count: 21
Fitted iterations: 100
```

`tests/test_features.py`: 27 tests đạt trong lần người thực hiện chạy.
Ruff báo đạt sau khi người thực hiện sửa thụt lề và thứ tự import.
Đây là xác nhận kỹ thuật về luồng dự báo, chưa phải kết quả độ chính xác.
Bản đầu tạo lại đặc trưng trên lịch sử trong mỗi ngày, có log tiến độ;
chưa đo thời gian chạy đầy đủ hoặc tối ưu hiệu năng. Điểm sai số được bổ sung
ở bước 4.4A dưới đây; chưa dùng test.

## 4.4. Validation và chọn số vòng học — hoàn thành trong ngân sách đã đặt

### 4.4A. Chấm điểm cửa sổ đầu tiên — đã có kết quả

Người thực hiện đã tạo `src/evaluation/score_lightgbm.py`, đọc bản CSV trong
artifact của run dự báo ở mục 4.3. Script chỉ dùng nhãn thật để chấm điểm
dự báo đã có, không đưa chúng trở lại đầu vào hoặc huấn luyện lại mô hình.

Quy tắc đã triển khai, nhất quán tuần 2:

- Ghép theo khóa `id + day`, kiểm tra đủ chuỗi/ngày và không trùng khóa.
- Tính riêng mỗi chuỗi: MAE, RMSE, sMAPE, RMSSE; sau đó lấy trung bình và
  trung vị qua các chuỗi.
- RMSSE dùng lịch sử d_1 đến mốc dự báo để tính mẫu số, cắt từ lần bán đầu
  tiên theo hàm đã có trong `src/evaluation/metrics.py`.
- Loại ngày đóng cửa khỏi thước đo theo `configs/split.toml`; validation
  đầu d_1606–d_1633 không có ngày 25/12.
- Ghi rõ số chuỗi tính được RMSSE; không giấu trường hợp mẫu số bằng 0.
- Ghi run chấm điểm mới trong MLflow, liên kết tới run dự báo nguồn bằng tag.

Kết quả người thực hiện gửi cho d_1606–d_1633, đủ 297 chuỗi và RMSSE xác định
cho cả 297 chuỗi (mỗi cặp dưới đây là trung bình / trung vị qua các chuỗi):

- MAE: 1,505456 / 0,998959.
- RMSE: 1,889309 / 1,252549.
- sMAPE (%): 124,540584 / 122,124075.
- RMSSE: 0,714618 / 0,664843.

Run dự báo nguồn: `a9ab59523715493da9a8415eb182ed5c`.
Run chấm điểm: `26e277e87f554fdca8ac6212ba943e3c`, trạng thái `FINISHED`.
Điểm từng chuỗi lưu ở
`reports/lightgbm/a9ab59523715493da9a8415eb182ed5c/per_series_window.csv`.
Người thực hiện đã chạy chung tests chấm điểm/lựa chọn mô hình: 4 tests đạt.

Điểm của một cửa sổ không được so trực tiếp với bảng baseline lấy trung bình
sáu cửa sổ. Chưa có cơ sở kết luận LightGBM hơn/kém baseline từ kết quả này.

### 4.4B. Chọn số vòng học bằng validation — đã có kết quả

Phương án ban đầu: kiểm tra sau mỗi 50 vòng học, tối đa 500 vòng; dừng sau
3 mốc liên tiếp không giảm RMSSE trung bình trên cửa sổ validation đầu.
Giữ mô hình tốt nhất trong các mốc đã kiểm tra. Các số 50/500/3 là lựa chọn
thí nghiệm, chưa được tối ưu, không phải tham số nghiệp vụ.

Người thực hiện đã đổi giới hạn `n_estimators` trong `configs/lightgbm.toml`
thành 500, thêm mục `[selection]` với `check_every=50`, `patience_checks=3`,
và tạo `src/forecast/select_lightgbm.py` cùng test tương ứng. Ruff đạt,
4 tests lựa chọn/chấm điểm đạt trong lần người thực hiện chạy.

Code dùng `LGBMRegressor.fit(..., init_model=...)` để học tiếp ở mỗi mốc.
Mỗi lần chỉ thêm một nhóm cây, tổng số vòng thực tế đọc bằng
`model.booster_.current_iteration()`. Mỗi mốc tạo một đối tượng mô hình mới
để giữ riêng mô hình tốt nhất; test kiểm tra mô hình được giữ có đúng
số vòng, không bị đổi thành mô hình cuối khi học tiếp.

Đây là dừng sớm theo mốc kiểm tra tự triển khai; không dùng callback dừng sớm
mặc định trên một bảng validation đã điền số bán thật vào lag. Mỗi mốc phải
dự báo đệ quy cả 28 ngày với cùng quy tắc giá/lịch, rồi chấm bằng `score_window`.
Các cây luôn học trên train đến d_1605; validation chỉ quyết định tiếp tục
hay dừng và chọn mô hình nào. Test vẫn được giữ riêng.

Chỉ chọn trong các mốc đã kiểm tra, không phải mọi số vòng có thể có. Nếu
chạm giới hạn 500 thì chưa kết luận dừng sớm do hết cải thiện; ghi rõ lý do dừng.
Điểm thấp nhất trên cửa sổ đã dùng để lựa chọn có thể lạc quan. Sau khi chọn
vẫn phải đánh giá đủ sáu cửa sổ, rồi cuối cùng mới dùng test khi khóa cấu hình.
Kết quả người thực hiện gửi và được đối chiếu với CSV lịch sử đã lưu:

- Run: `7f9a50c48080420ebd1eb805aa62249d`, trạng thái `FINISHED`.
- Đã kiểm tra 10 mốc: 50, 100, ..., 500 cây; RMSSE giảm ở mọi mốc.
- RMSSE ở 50 cây: 0,775881; 100 cây: 0,714618; 500 cây: 0,696867361090941.
- Số vòng được chọn: 500; lý do dừng: `max_rounds`.
- Quy tắc chờ 3 mốc không cải thiện chưa kích hoạt trong lần chạy thật này.

Kết luận: 500 cây tốt nhất trong phạm vi đã kiểm tra, nằm ngay biên ngân sách.
Chưa chứng minh điểm đã hội tụ hoặc 500 là số vòng tối ưu. Giữ lựa chọn này
cho bước so sánh kế tiếp; không tự mở rộng giới hạn dựa trên điểm test.

Các file được lưu tại `reports/lightgbm/7f9a50c48080420ebd1eb805aa62249d/`:
`selection_history.csv`, `per_series_window.csv`, `lightgbm_valid_first_window.csv`.
Mô hình và danh sách nhóm được lưu trong artifacts của run MLflow.
Trước commit mục 4, trợ lý đã chạy kiểm tra toàn repo: 85 tests đạt, Ruff đạt.

## Mục 5: so sánh trên sáu cửa sổ validation — hoàn thành

Quy trình đã được người thực hiện chạy và xác nhận:

- Giữ các tham số đã chọn, dùng 500 cây ở cả sáu cửa sổ.
- Fit lại mô hình tại từng mốc bằng dữ liệu đến hết ngày trước cửa sổ,
  mở rộng lịch sử sau mỗi 28 ngày, giống lịch refit của baseline tuần 2.
  Danh sách nhóm cũng chỉ học từ phần lịch sử được phép tại mốc đó.
- Bên trong mỗi cửa sổ, dự báo đệ quy, không dùng số bán thật để cập nhật
  trước khi cửa sổ kết thúc. Giá tương lai tiếp tục giữ ở mức cuối đã biết.
- Tính thước đo cho từng chuỗi/từng cửa sổ, lấy trung bình qua sáu cửa sổ
  cho mỗi chuỗi, rồi lấy trung bình/trung vị qua các chuỗi, đúng baseline.
- Dùng lại `reports/baselines/per_series_valid.csv` và kiểm tra cùng tập chuỗi,
  mốc ngày, độ dài cửa sổ. Không chạy lại hoặc chỉnh cấu hình baseline theo test.
- Cửa sổ đầu đã được dùng chọn số vòng nên điểm validation tổng hợp có thể
  lạc quan; chưa phải đánh giá cuối trên test. Chưa tăng ngân sách số cây.

## Các bước tiếp theo

- [x] 4.1. Chuẩn bị `X`, `y` và danh sách nhóm chỉ từ train.
- [x] 4.2. Huấn luyện kiểm tra lần đầu và xác nhận mô hình được lưu trong MLflow.
- [x] 4.3. Tạo luồng dự báo 28 ngày dùng dự báo làm lịch sử; kiểm tra lại rò rỉ
  và cách áp dụng danh sách nhóm đã học từ train.
- [x] 4.4. Triển khai cơ chế dừng sớm theo mốc, chọn số vòng trong ngân sách
  500 cây; lần chạy thật dừng do chạm ngân sách, chưa do hết cải thiện.

Mục 4 hoàn thành với các giới hạn đã ghi. So sánh đủ sáu cửa sổ với baseline
thuộc mục 5. Người thực hiện đã yêu cầu commit mốc mục 4 trước khi chuyển mục.

## Kết quả mục 5

Run `08e87e3b435c4aa6a8cb5809d2ab07fb` hoàn tất: 6 cửa sổ, 49.896 dòng dự báo, 8 cách. Ruff đạt và 30 tests so sánh/đặc trưng đạt trong lần người thực hiện chạy.

LightGBM 500 cây refit có RMSSE trung bình/trung vị 0,694558/0,664376, MAE 1,528189/0,988386, RMSE 1,930814/1,264500, sMAPE (%) 127,237040/127,694041. Theo RMSSE trung bình, LightGBM đứng thứ ba sau ETS (0,673912) và trung bình trượt 28 ngày (0,677792). Chưa có phép đo độ chắc chắn cho chênh lệch. Đây là kết quả validation, không phải điểm test.

File kết quả: `reports/lightgbm/08e87e3b435c4aa6a8cb5809d2ab07fb/`. Chưa kết luận chi phí tồn kho hoặc mức đáp ứng đơn vì chưa có mô phỏng kho.

## Mục 6: tầm quan trọng đặc trưng — hoàn thành

Chạy `python -m src.evaluation.feature_importance --run-id 08e87e3b435c4aa6a8cb5809d2ab07fb`.
Run importance: `48344fac031d49f790ce767f5b20c8fe`. Đọc lại sáu mô hình đã lưu,
không fit lại hoặc dùng test. Kết quả ở thư mục `importance/` trong thư mục run so sánh:
`per_window.csv`, `summary.csv`, `gain_importance.png`.

Gain là tổng mức giảm hàm mất mát train tại các phép chia dùng đặc trưng;
split là số lần dùng đặc trưng để chia. Chuẩn hóa tổng gain về 100% riêng từng
mô hình, sau đó lấy trung bình đều qua sáu mô hình để mô hình có tổng gain lớn
không tự nhận trọng số cao hơn. Độ lệch chuẩn/min/max trong CSV chỉ mô tả sự
khác nhau giữa mô hình; không phải khoảng tin cậy vì lịch sử train chồng lấn.

Kết quả đo: `sales_roll_mean_7` chiếm 84,9828% gain chuẩn hóa trung bình
(min 84,4058%, max 85,5606%). Tiếp theo là `sell_price_lag_1` 2,6826%,
`sales_lag_14` 2,4584%, `day_of_week` 2,2053%, `sales_lag_7` 2,0181%.
Theo split, `sales_lag_28` được dùng trung bình 2.845,67 lần/mô hình dù gain
chỉ 1,5698%, cho thấy thứ hạng phụ thuộc cách định nghĩa importance.
`state_id`, `event_name_2`, `event_type_2` không được dùng trong sáu mô hình này.

Nhận xét: mô hình dựa mạnh vào mức bán gần đây theo thước đo gain trên train.
Không diễn giải 84,98% là độ chính xác hoặc mức cải thiện trên validation.
Không suy ra giá gây thay đổi nhu cầu hoặc cờ bang/sự kiện không hữu ích nói chung.
Các đặc trưng tương quan có thể chia sẻ hoặc thay thế tín hiệu; một cửa hàng
mỗi bang còn làm `state_id` trùng thông tin với `store_id`. Chưa làm ablation
hoặc đo tầm quan trọng bằng thay đổi sai số dự báo đệ quy trên validation.

Đã sửa cách nạp artifact text Windows: đọc thành chuỗi chuẩn hóa CRLF về LF
trước khi khởi tạo Booster, giữ nguyên file gốc. 7 tests importance đạt,
gồm test lưu/nạp CRLF giữ nguyên dự báo và gain. Biểu đồ đã kiểm tra trực quan.
Tham khảo định nghĩa: https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html#lightgbm.Booster.feature_importance

## Mục 7: báo cáo — hoàn thành

Báo cáo tổng hợp ở [reports/week3_lightgbm.md](../reports/week3_lightgbm.md),
gồm đặc trưng, mô hình, cách chọn số vòng, so sánh sáu cửa sổ, importance,
nguồn kết quả và giới hạn. Tuần 3 hoàn tất; giữ kết luận LightGBM chưa vượt
hai baseline mạnh nhất theo RMSSE validation. Tuần 4 chưa được triển khai.
