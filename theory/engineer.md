## IO Engineering Methods

**This chapter is for better I/O handling.**

### Tip 1: The Data Iterator

For large datasets it is not feasible to pre-load the entire dataset first all into our memory. What is needed is a mechanism by which we can quickly and efficiently stream data directly from the source. **MX-Net** has provided a method as follows:

>Data iterator is the mechanism by which we feed input data into an MX-Net training algorithm and they are very simple to initialize and use and are optimized for speed. During training, we typically process training samples in **small batches** and over the entire training lifetime will end up processing each training example multiple times. 

