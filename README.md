# Overview
An implementation of GPU ray tracing on NVIDIA RTX 2080Ti. The origin CPU code is in Peter Shirley's
[Ray Tracing in One Weekend book](https://raytracing.github.io/books/RayTracingInOneWeekend.html).
Same image in the CPU version takes about 1559.08 seconds to render and the GPU version takes about 89.65 seconds. 
It results in a 17.4x speed up. And the optimized version further results in a 2.64x speed up.  
  
The configurations used to render the final image are: 1200*800 resolution, 200 samples per pixel, 50 ray bounces.
  
Project report: [Google Doc Link](https://docs.google.com/document/d/1Vn1uWVYVuFz_-aaSXAt4Z4GZcK2T2EoMx00ITAH-ADI/edit)

# Poster
![alt text](https://github.com/guiqi134/CS6235_GPU_RayTracing/blob/master/CS6235_Poster.png?raw=true)
