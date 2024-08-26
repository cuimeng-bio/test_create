# 安装并加载必要的包
library(ggplot2)
library(dplyr)
filepath="E:\\desktop\\blast.txt"
file = read.delim(filepath,sep="\t",header=T,stringsAsFactor=F)
file = file%>%filter(Query_ID%in%c("1 Genus species strain strain"))
# 生成示例数据
# 绘制比对图
ggplot(file) +
  geom_segment(aes(x = Subject_Start, y = Query_Start, xend = Subject_End, yend = Query_End), color = "purple") +
  geom_point(aes(x = Subject_Start, y =Query_Start ), color = "purple") +
  geom_point(aes(x = Subject_End, y = Query_End), color = "purple") +
  scale_y_continuous(breaks = c(1, 2), labels = c("Query", "Subject")) +
  xlab("Subject") + ylab("Query") +
  ggtitle("Dot Plot of Sequence Alignment") +
  theme_minimal()

# 显示图形
ggsave("dot_plot_example.png")
