#include <iostream>
#include <cstdlib>
#include <cfloat>
#include <curand_kernel.h>

#include "includes/common.h"
#include "includes/hittable_list.h"
#include "includes/sphere.h"
#include "includes/camera.h"
#include "includes/material.h"

using namespace std;

const size_t IMAGE_WIDTH = 1200;
const size_t IMAGE_HEIGHT = 800;
const bool OUTPUT = true;

// val -> the return value of CUDA calls
#define checkCudaError(val) checkError( (val), #val)
void checkError(cudaError_t result, const char* func)
{
	if (result != cudaSuccess)
	{
		cerr << "CUDA error: " << cudaGetErrorString(result) << " at " << __FILE__
			<< ", line " << __LINE__ << ", func = " << func << endl;
		// reset CUDA device before exiting
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

__device__ color rayColor(const ray& r, hittable** d_world, curandState* state, int maxDepth)
{
	ray currentRay = r;
	color currentAttenuation = color(1.0f, 1.0f, 1.0f);
	for (int i = 0; i < maxDepth; i++)
	{
		hitRecord rec;
		if ((*d_world)->hit(currentRay, 0.001, FLT_MAX, rec))
		{
			ray scattered;
			color attenuation;
			if (rec.mat_ptr->scatter(currentRay, rec, attenuation, scattered, state))
			{
				currentRay = scattered;
				currentAttenuation *= attenuation;
			}
			else
				return color(0.0f, 0.0f, 0.0f);
		}
		else
		{
			vec3 unitDir = unit_vector(currentRay.direction());
			float t = 0.5f * (unitDir.y() + 1.0f);
			color clr = (1.0f - t) * color(1.0f, 1.0f, 1.0f) + t * color(0.5, 0.7, 1.0);
			return currentAttenuation * clr;
		}
	}
	return color(0.0f, 0.0f, 0.0f);
}

__global__ void render(color* framebuffer, int width, int height, int samplesPerPixel, curandState* state, 
	hittable** d_world, camera** d_camera, int maxDepth)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >= width || j >= height) return;

	int index = j * width + i;

	// construct the ray
	color pixelColor(0.0f, 0.0f, 0.0f);
	for (int s = 0; s < samplesPerPixel; s++)
	{
		float u = float((i + random_float(&state[index]))) / float(width - 1);
		float v = float((j + random_float(&state[index]))) / float(height - 1);
		ray r = (*d_camera)->get_ray(u, v, &state[index]);
		pixelColor += rayColor(r, d_world, &state[index], maxDepth);
	}
	pixelColor /= samplesPerPixel;


	// gamma correction = 2.0f
	pixelColor[0] = sqrtf(pixelColor[0]);
	pixelColor[1] = sqrtf(pixelColor[1]);
	pixelColor[2] = sqrtf(pixelColor[2]);
	framebuffer[index] = pixelColor;
}

__global__ void setupRender(curandState* state, int width, int height)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i >= width || j >= height) return;
	int index = j * width + i;
	// Each thread gets same seed, a different sequence number, no offset 
	curand_init(1998, index, 0, &state[index]);
}

__global__ void randInit(curandState* state)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
		curand_init(1998, 0, 0, state);
}

__global__ void randomScene(hittable** d_world, hittable** d_objects, camera** d_camera, curandState* state)
{
	material* materialGround = new lambertian(color(0.5, 0.5, 0.5));
	d_objects[0] = new sphere(point3(0, -1000, 0), 1000, materialGround);
	int index = 1;

	// random small spheres
	for (int a = -11; a < 11; a++)
	{
		for (int b = -11; b < 11; b++, index++)
		{
			float material = random_float(state);
			point3 center(a + 0.9 * random_float(state), 0.2, b + 0.9 * random_float(state));

			color albedo;
			if (material < 0.8)
			{
				// diffuse
				albedo = color::random(state) * color::random(state);
				d_objects[index] = new sphere(center, 0.2, new lambertian(albedo));
			}
			else if (material < 0.95)
			{
				// metal
				albedo = color::random(state, 0.5, 1);
				float fuzz = random_float(state, 0.5, 1);
				d_objects[index] = new sphere(center, 0.2, new metal(albedo, fuzz));
			}
			else
			{
				// glass
				d_objects[index] = new sphere(center, 0.2, new dielectric(1.5));
			}
		}
	}

	// 3 big spheres
	d_objects[index] = new sphere(point3(0, 1, 0), 1.0, new dielectric(1.5));
	d_objects[++index] = new sphere(point3(-4, 1, 0), 1.0, new lambertian(color(0.4, 0.2, 0.1)));
	d_objects[++index] = new sphere(point3(4, 1, 0), 1.0, new metal(color(0.7, 0.6, 0.5), 0.0));

	*d_world = new hittable_list(d_objects, 488);
	
	// camera
	point3 lookfrom(13, 2, 3);
	point3 lookat(0, 0, 0);
	vec3 vup(0, 1, 0);
	float dist_to_focus = 10.0;
	float aperture = 0.1;
	*d_camera = new camera(lookfrom, lookat, vup, 35, 4.0f / 3.0f, aperture, dist_to_focus);
}

__global__ void createWorld(hittable** d_world, hittable** d_objects, camera** d_camera, int numObjects)
{
	// execuate only once
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		point3 lookfrom(0, 3, 3);
		point3 lookat(0, 0, -1);
		vec3 vup(0, 0, -1);
		float dist_to_focus = (lookfrom - lookat).length();
		float aperture = 2.0;

		material* material_ground = new lambertian(color(0.8, 0.8, 0.0));
		material* material_center = new lambertian(color(0.1, 0.2, 0.5));
		material* material_left = new dielectric(1.5);
		material* material_right = new metal(color(0.8, 0.6, 0.2), 1.0);

		*d_objects = new sphere(point3(0.0, -100.5, -1.0), 100.0, material_ground);
		*(d_objects + 1) = new sphere(point3(0.0, 0.0, -1.0), 0.5, material_center);
		*(d_objects + 2) = new sphere(point3(-1.0, 0.0, -1.0), 0.5, material_left);
		*(d_objects + 3) = new sphere(point3(1.0, 0.0, -1.0), 0.5, material_right);
		*(d_objects + 4) = new sphere(point3(-1.0, 0.0, -1.0), -0.45, new dielectric(1.5));
		*d_world = new hittable_list(d_objects, numObjects);
		*d_camera = new camera(lookfrom, lookat, vup, 30, 4.0f/3.0f, aperture, dist_to_focus);
	}
}

__global__ void deleteWorld(hittable** d_world, hittable** d_objects, camera** d_camera, int numObjects)
{
	// execuate only once
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		for (int i = 0; i < numObjects; i++)
		{
			delete ((sphere*)d_objects[i])->mat_ptr;
			delete d_objects[i];
		}
		delete* d_world;
		delete* d_camera;
	}
}

int main(int argc, char* args[])
{
	size_t num_pixels = IMAGE_WIDTH * IMAGE_HEIGHT;
	size_t framebufferSize = num_pixels * sizeof(vec3);
	int samplesPerPixel = 200;
	int maxDepth = 50;
	int numObjects = 488;
	
	// allocate framebuffer to unified memory
	color* framebuffer;
	checkCudaError(cudaMallocManaged((void**)&framebuffer, framebufferSize));

	// allocate camera
	camera** d_camera;
	checkCudaError(cudaMalloc((void**)&d_camera, sizeof(camera*)));

	// allocate random states
	curandState* d_state;
	checkCudaError(cudaMalloc((void**)&d_state, num_pixels * sizeof(curandState)));
	curandState* d_state2;
	checkCudaError(cudaMalloc((void**)&d_state2, sizeof(curandState)));

	// allocate world and objects
	hittable** d_world;
	size_t d_worldSize = sizeof(hittable*);
	checkCudaError(cudaMalloc((void**)&d_world, d_worldSize));
	hittable** d_objects;
	size_t d_objectsSize = numObjects * sizeof(hittable*);
	checkCudaError(cudaMalloc((void**)&d_objects, d_objectsSize));

	// setup random state for randomScene
	randInit<<<1, 1>>>(d_state2);
	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceSynchronize());

	// setup randomScene
	randomScene<<<1, 1>>>(d_world, d_objects, d_camera, d_state2);
	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceSynchronize());

	// configure parameters
	int blockDimX = 8, blockDimY = 8;
	dim3 dimBlock(blockDimX, blockDimY);
	dim3 dimGrid((IMAGE_WIDTH + blockDimX - 1) / blockDimX, (IMAGE_HEIGHT + blockDimY - 1) / blockDimY);

	// setup random state for render
	setupRender<<<dimGrid, dimBlock>>>(d_state, IMAGE_WIDTH, IMAGE_HEIGHT);
	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceSynchronize());

	// invoke render
	render<<< dimGrid, dimBlock >>>(framebuffer, IMAGE_WIDTH, IMAGE_HEIGHT, samplesPerPixel, d_state, d_world, d_camera, maxDepth);
	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceSynchronize());

	// using command line param to output rendered PPM image
	if (OUTPUT)
	{
		std::cout << "P3\n" << IMAGE_WIDTH << " " << IMAGE_HEIGHT << "\n255\n";
		for (int j = IMAGE_HEIGHT - 1; j >= 0; j--)
		{
			for (int i = 0; i < IMAGE_WIDTH; i++)
			{
				int index = j * IMAGE_WIDTH + i;
				int ir = int(255.99 * framebuffer[index].x());
				int ig = int(255.99 * framebuffer[index].y());
				int ib = int(255.99 * framebuffer[index].z());
				std::cout << ir << " " << ig << " " << ib << endl;
			}
		}
	}

	// clean up
	deleteWorld<<<1, 1>>>(d_world, d_objects, d_camera, numObjects);
	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceSynchronize());
	checkCudaError(cudaFree(d_world));
	checkCudaError(cudaFree(d_objects));
	checkCudaError(cudaFree(d_camera));
	checkCudaError(cudaFree(d_state));
	checkCudaError(cudaFree(framebuffer));

	cudaDeviceReset();

	return 0;
}