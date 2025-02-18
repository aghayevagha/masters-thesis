# Identity-Based Encryption (IBE) Implementation

This project focuses on implementing an **Identity-Based Encryption (IBE)** scheme using the **Dual Regev** approach. It includes key generation, encryption, and decryption functionalities while leveraging **Discrete Gaussian Sampling** and **Gadget matrix techniques**.

## Project Structure

- **`main.py`**  
  - Allows you to send a message and attempt decryption.  
  - **Note:** Decryption is not working correctly yet.  

- **`IBE.py`**  
  - Implements key generation for **Master Public Key (MPK)** and **Master Secret Key (MSK)**.  
  - Uses the **GPV scheme** to generate a uniform matrix sampled from a **Discrete Gaussian distribution**.  
  - The MPK is reduced to a **Gadget matrix**, which is publicly known and helps in inverting the **Short Integer Solution (SIS) problem**.

- **`utils.py`**  
  - Handles binary representation and Gadget matrix generation.

- **`user.py`**  
  - Not yet implemented.  
  - Requires a **hash function** to convert a user's **ID to a vector**, which is still under development.  

## Current Issues

1. **Encryption-Decryption Bug:**  
   - The decryption process fails, possibly due to an error in the **Dual Regev scheme** implementation.  
   - Debugging the issue to identify what is missing.

2. **User ID to Vector Conversion:**  
   - Need to use a hash function and divide it into shorter pieces, and map  **user's ID to a vector**.

## Future Plans

1. Validate the **correctness** of the DGS method.  


