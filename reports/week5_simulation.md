# Tuần 5 — Bộ mô phỏng tồn kho và lần chạy M5

## Đoạn mô tả dùng trong báo cáo đồ án

Đồ án đã xây dựng bộ mô phỏng tồn kho theo ngày cho một cặp sản phẩm–cửa
hàng. Mỗi ngày, mô phỏng nhận các đơn đến hạn vào đầu ngày, đáp ứng nhu
cầu bằng lượng hàng có sẵn, ghi nhận phần bán mất và xét đặt thêm vào
cuối ngày. Người thực hiện chọn giả định **lost sales**, theo đó phần
nhu cầu chưa đáp ứng được ghi là lượng bán mất của ngày đó, ngày tiếp
theo xử lý nhu cầu mới. Đơn đặt cuối ngày t được nhận đầu ngày t + L,
với L là số ngày lịch giả định. Bộ mô phỏng theo dõi tồn kho và từng
lượng hàng đang về theo ngày đến, đồng thời cho phép truyền hàm chính
sách nhận trạng thái cuối ngày để quyết định lượng đặt. Lần chạy minh
họa trên chuỗi M5 `FOODS_2_148_WI_1_evaluation`, từ 22/06 đến 19/07/2015,
có tổng nhu cầu mô phỏng 19 đơn vị, đáp ứng 19, bán mất 0 và tồn cuối
kỳ 1 đơn vị. Kết quả sử dụng **giả định** doanh số M5 đại diện nhu cầu,
thời gian chờ 3 ngày, mức đặt tới 4 đơn vị và tồn ban đầu 4 đơn vị.
Các trường hợp tính tay và kiểm thử tự động đã được đối chiếu, toàn
repo đạt 190 tests trong lần chạy người thực hiện xác nhận.

## Quy tắc mô phỏng và giả định

Các lượng nhu cầu, tồn, đặt và nhận được biểu diễn bằng số nguyên không
âm. Đơn nhận vào đầu ngày có thể dùng để đáp ứng nhu cầu trong ngày.
Nhà cung cấp được giả định giao đủ lượng đã đặt sau L ngày lịch. Tồn
ban đầu được tính trước lần nhận đầu tiên, các đơn đang về được truyền
riêng theo ngày đến. Cuối kỳ, kết quả giữ các đơn sẽ về sau cửa sổ chạy.

Vị thế tồn kho bằng tồn có sẵn cộng hàng đang về trong cách xử lý lost
sales. Chính sách nhận vị thế trước khi đặt đơn mới hôm nay. Nhật ký
ghi vị thế sau khi đặt để người đọc đối chiếu trạng thái chuyển sang
ngày tiếp theo. Hàm chính sách được truyền lịch sử từ đầu mô phỏng đến
hết ngày hiện tại. Lịch sử này sử dụng đầy đủ nhu cầu đầu vào theo
**giả định quan sát được nhu cầu quá khứ** đã ghi trong dự án.

Ngày 25/12 được giả định đóng cửa. Mô phỏng giữ ngày lịch, dùng nhu cầu
0 và dừng xét đặt thêm, đồng thời tiếp tục nhận các đơn đến hạn và đếm
thời gian chờ. Quy tắc này được kiểm tra bằng dữ liệu giả lập. Cửa sổ
M5 của lần chạy minh họa có 0 ngày đóng cửa. Các lựa chọn và lý do được
ghi tại mục 28–30 của file giả định. [Giả định của dự án](../docs/assumptions.md)

## Cấu hình và kết quả minh họa

Chuỗi được giữ từ ví dụ mức bán giữa theo train của tuần 4. Cửa sổ chạy
là d_1606–d_1633. Mức đặt tới được xác định trước cửa sổ bằng
`ceil(trung bình ngày mở cửa của 28 ngày train gần nhất * (L + 1))`.
Trung bình đo từ d_1578–d_1605 bằng 1 đơn vị/ngày. Chọn **L = 3 ngày**
cho demo dẫn đến mức đặt tới 4 đơn vị. Tồn ban đầu bằng 4 và các đơn
đang về ban đầu rỗng. Cuối mỗi ngày mở cửa, chính sách đặt
`max(4 - vị thế tồn kho, 0)`. Công thức mức đặt và trạng thái đầu là
**lựa chọn minh họa** để kiểm tra mô phỏng. [Cấu hình demo](../configs/inventory_demo.toml)

Lần chạy người thực hiện gửi có tổng lượng đặt 19 đơn vị. Trong kỳ nhận
16 đơn vị, đáp ứng 19 từ tổng lượng có sẵn là `4 + 16`, nên còn 1 đơn
vị cuối kỳ. Ba đơn vị đang về còn lại gồm 2 đơn vị đến d_1635 và 1 đơn
vị đến d_1636. Doanh số quan sát cung cấp đầu vào nhu cầu, còn lượng đáp
ứng, bán mất và tồn phụ thuộc các giả định vận hành đã nêu. Nguồn dùng
để chốt báo cáo là lần chạy `d38860b57d5740e3b2ad561ed53dd68f`.
[Nhật ký theo ngày](simulation/week5_demo/d38860b57d5740e3b2ad561ed53dd68f/daily.csv),
[tham số và tổng lượng](simulation/week5_demo/d38860b57d5740e3b2ad561ed53dd68f/run_info.json)

Biểu đồ gồm tồn cuối ngày và hàng đang về, lượng đáp ứng và bán mất,
cùng lượng nhận và lượng đặt. Số bán mất của cửa sổ này bằng 0, các test
tính tay có thêm những ngày thiếu hàng để kiểm tra nhánh lost sales.
[Biểu đồ lần chạy đã xác nhận](simulation/week5_demo/d38860b57d5740e3b2ad561ed53dd68f/inventory.png)

## Kiểm tra và khả năng chạy lại

Đã bổ sung 34 ca kiểm thử cho lõi mô phỏng và giao diện chính sách, cùng
6 ca cho bước nối dữ liệu và xuất hình. Các ca kiểm tra gồm thời điểm
nhận hàng, lượng bán mất, đơn về sau cuối kỳ, trạng thái trước đặt hàng,
lịch sử chính sách, tiếp tục chạy từ trạng thái cuối, đầu vào sai và
quy tắc ngày đóng cửa. Tổng số test toàn repo là 190. Output người thực
hiện xác nhận đạt `190 passed, 1 warning in 21.41s`. Cảnh báo deprecation
đến từ SQLAlchemy trong MLflow. Ruff đạt trên các module và test mô
phỏng đã bổ sung. [Test lõi mô phỏng](../tests/test_inventory.py),
[test nối dữ liệu và xuất hình](../tests/test_inventory_demo.py)

Tại thư mục gốc với `.venv` đã kích hoạt, chạy các lệnh sau.

```powershell
python -m pytest
python -m src.simulation.run_demo
```

Mỗi lần chạy demo tạo thư mục riêng dưới `reports/simulation/week5_demo/`,
gồm CSV nhật ký, PNG biểu đồ và JSON tham số hiệu lực. `conftest.py` cấp
thư mục tạm riêng cho mỗi phiên pytest để xử lý xung đột quyền thư mục
tạm giữa tài khoản Windows và tài khoản sandbox. [Hướng dẫn tuần 5](../docs/week5_simulation.md)

## Phạm vi kết quả và việc tiếp theo

Tuần 5 hoàn thành bộ mô phỏng một sản phẩm, giao diện chính sách, các
test tính tay và một ví dụ M5 có biểu đồ. Kết quả đáp ứng đủ 19 đơn vị
phản ánh một chuỗi trong 28 ngày với tồn khởi đầu và L đã chọn. Độ nhạy
với trạng thái đầu, thời gian chờ và mức đặt cần được kiểm tra trong
phần thực nghiệm tiếp theo. Giả định doanh số đại diện nhu cầu và hạn
chế chọn tập con bằng thông tin toàn giai đoạn tiếp tục được giữ trong
phạm vi diễn giải kết quả.

Tuần 6 bổ sung chi phí lưu kho, định nghĩa thước đo mức đáp ứng theo
lượng hàng và các cách đặt hàng để chuẩn bị so sánh ở cùng mức đáp ứng.
Cấu hình minh họa tuần 5 là điểm kiểm tra vận hành cho các bước đó.
