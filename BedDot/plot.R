
# Read the dataset from the CSV file
data <- read_csv("/Users/racheldong/Desktop/Re_Ke/BedDot/df_train_SBP_100_130_single.csv")

Time <- rep(0:100, time = 18907)
Signal <- as.matrix(data[,1:101])
Signal_1d <- c(Signal)

# Assuming you have a data frame 'data' with 'area' and 'price' columns
# and 'new_x' contains the new input values of 'area'

# Fit kernel smooth to the original data
smoothed_fit <- density(Signal_1_d, kernel = "gaussian")

# Predict y-values (fitted prices) for new input x-values
new_x <- c(0:100)
predicted_y <- approx(smoothed_fit$y, smoothed_fit$x, xout = new_x, rule = 2)$y

# Plot the original data points as lines
png(file = "/Users/racheldong/Downloads/compare.png", width=640, height=480)
plot(dt$Time, dt$Signal, pch = 19,  
     col = "blue", 
     cex = 0.25)
plot(smoothed_fit$x, smoothed_fit$y, col = "red", lwd = 2, type = "l")
points(new_x, predicted_y, pch = 19, col = "green")
legend("topright", legend = c("Original Data", "Kernel Smooth", "Predicted Values"), col = c("blue", "red", "green"), pch = c(19, NA, 19))
dev.off()


true <- read.csv("/Users/racheldong/Desktop/Re_Ke/BedDot/simulator/true_singal_115.csv")
plot(new_x, true[90:190,], type = 'l')

png(file = "/Users/racheldong/Downloads/signal.png", width=640, height=480)

# Create a plot with an empty canvas
plot(1, type = 'n', 
     xlim = c(0, 100), 
     ylim = c(min(data), max(data[, 1:101])), 
     xlab = "Time", ylab = " ", 
     main = "True Single Curve v.s. Spline Fitted Curve (K=30)")

# Loop through each row and plot the signals
for (i in 1:nrow(data)) {
  signal <- data[i, 1:101]
  lines(0:100, signal, col = i, lwd = 0.5, alpha = 0.5)
}
lines(mod.ss, lwd = 3, col = 'red')
lines(new_x, true[90:190,], lwd = 3, col = 'blue')

dev.off()
# # Add a legend to distinguish the different signals
# legend("topright", legend = 1:nrow(data), col = 1:nrow(data), lty = 1, title = "Signal")

Signal_vector <- unlist(apply(Signal, 1, c))
Signal_1_d <- c(Signal_vector)
dt <- data.frame(Time = Time,
                 Signal = Signal_1_d)
png(file = "/Users/racheldong/Downloads/scatter.png", width=640, height=480)
plot(dt$Time, dt$Signal, pch = 19,  
     col = "blue", 
     cex = 0.25)
dev.off()



library(npreg)
x = dt$Time
y = dt$Signal

mod.ss <- ss(x, y, nknots = 35)
mod.ss

# Assuming your dataset is named 'my_data' and you want to save it to a CSV file named 'my_dataset.csv'
write.csv(mod.ss$y, file = "/Users/racheldong/Downloads/spline.csv", row.names = FALSE)




png(file = "/Users/racheldong/Downloads/compare_spline.png", width=640, height=480)
plot(dt$Time, dt$Signal, pch = 19,  
     col = "blue", 
     cex = 0.25,
     ylab = '',
     xlab = 'Time',
     main = "Spline with 95% Confidence Band (K=30)")
lines(mod.ss, lwd = 3, col = 'orange')
dev.off()









