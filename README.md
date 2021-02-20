# Kaggle-Favorita
# Decription
Corporacion Favorita grocery sales prediction competition
# Data
Source: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data
About:
 1. Train: bao gồm các cột là id, date, store_nbr, item_nbr, unit_sales, onpromotion
  +Cột id là số thứ tự
  +Cột date là ngày tháng
  +Cột unit_sales là đại diện cho lợi nhuận của mặt hàng cụ thể
  +Cột onpromotion là cho biết items_nbr đó có được khuyến mãi trong một ngày cụ thể ở cột date và cửa hàng cụ thể ở cột store_nbr.
 2. Test: bao gồm các cột id, date, store_nbr, item_nbr, onpromotion
  +Các cột data, store_nbr, item_nbr kết hợp để dự đoán cùng với thông tin khuyến mãi ở cột onpromotion
 3. Stores: bao gồm các cột store_nbr, city, state, type, cluster
  +Cột city tên thành phố ở bang cụ thể ở cột state
  +Cột cluster là mọt nhóm các của hàng tương tự
  +Cột type là các loại cửa hàng
 4. Items: bao gồm các cột item_nbr, family, class, perishable
  +Cột item_nbr: mã mặt hàng
  +Cột family, class: họ và lớp của mặt hàng
  +Cột perishable: cho biết mặt hàng có dễ hư hỏng hay không, ứng với 0 là không dễ hư hỏng và 1 là dễ bị hư hỏng
  +LƯU Ý: Các mục được đánh dấu là perishable có trọng số điểm là 1.25;   nếu không, trọng lượng là 1.0.
 5. Transaction: bao gồm các cột date, store_nbr, transaction
  +Cột transaction là số lượng giao dịch mỗi ngày kết hợp với store_nbr
 6. Oil: bao gồm các cột là date, dcoilwtico
  +Cột dcoilwtico: giá dầu hằng ngày
  +Cột date: tháng/ngày/năm
 7. Holidays_events: bao gồm các ngày lễ, sự kiện và siêu dữ liệu
  +LƯU Ý: Đặc biệt chú ý đến cột transferred . Một ngày lễ transferred chính thức rơi vào ngày dương lịch đó, nhưng đã được chính phủ dời sang một ngày khác. Một transferred ngày giống như một ngày bình thường hơn là một ngày lễ. Để tìm ngày mà nó đã thực sự ăn mừng, nhìn cho hàng tương ứng ở đâu type là Transfer. Ví dụ: ngày lễ Independencia de Guayaquil được chuyển từ 2012-10-09 sang 2012-10-12, có nghĩa là nó được tổ chức vào 2012-10-12. Những ngày được loại Bridge là những ngày bổ sung được thêm vào một kỳ nghỉ (ví dụ: để kéo dài thời gian nghỉ trong một ngày cuối tuần dài). Những điều này thường được tạo thành bởi loại Work Day ngày thường không được lên lịch làm việc (ví dụ: thứ bảy) nhằm mục đích hoàn vốn cho Bridge.
  +Additional ngày lễ là những ngày được thêm vào một ngày lễ theo lịch thông thường, chẳng hạn như thường xảy ra vào khoảng lễ Giáng sinh (biến đêm Giáng sinh thành một ngày lễ).
# Data Exploration
Các bước thực hiện:
_Number of rows/columns
_Check for missing values
_Histogram plot
_Summary All analysis should be applied to whole data set
Links:
_Stores
_Holiday
_Transaction
_Oil
_Items




