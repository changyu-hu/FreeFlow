// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/common/CudaCommon.cuh"

#include "FSI_Simulator/core/SolidGeometryProxy_Device.cuh"
#include "FSI_Simulator/core/FsiCoupling.hpp"

#include "FSI_Simulator/lbm/LbmDataTypes.cuh"
#include "FSI_Simulator/lbm/LbmUtilsFunc.cuh"
#include "FSI_Simulator/lbm/LbmConstants.hpp"

namespace fsi
{

	namespace lbm
	{

		namespace LbmD2Q9
		{
			extern __constant__ float c_w[Q];
			extern __constant__ float c_ex[Q];
			extern __constant__ float c_ey[Q];
			extern __constant__ float c_cs2;
			extern __constant__ int c_inv[Q];
		}

		__global__ void mrSolver2DComputeOccupancyFlagKernel_v3(uint32_t *hit_index, SolidGeometryProxy_Device<2> solid_proxy, int sample_x, int sample_y, float dx, int ecnt)
		{
			int idx = threadIdx.x + blockDim.x * blockIdx.x;
			int xid = idx / 8;
			int l = idx % 8 + 1;
			int x = xid / 9;
			int id = xid % 9;
			int x_id = id / 3;
			int y_id = id % 3;
			if (x >= 0 && x < ecnt)
			{
				glm::vec2 bmin, bmax;

				solid_proxy.getBoundingBox(x, bmin, bmax);

				bmin /= dx;
				bmax /= dx;

				for (int i = int(bmin[0]) - 1 + x_id; i <= int(bmax[0]) + 1; i += 3)
				{
					if (i >= 0 && i < sample_x)
					{
						for (int j = int(bmin[1]) - 1 + y_id; j <= int(bmax[1]) + 1; j += 3)
						{
							if (j >= 0 && j < sample_y)
							{
								int ind = j * sample_x + i;
								uint32_t *hitIndex = hit_index + ind * 9;
								//*hitIndex = 1;
								// //mlflow[0].hitIndex[ind * 27] = 1;
								IntersectionInfo info;

								glm::vec2 pos1(i, j), pos2(i - LbmD2Q9::c_ex[l], j - LbmD2Q9::c_ey[l]);
								pos1 *= dx;
								pos2 *= dx;
								info.element_index = x;
								info.depth = 3.;
								solid_proxy.intersectAtIndex(pos1, pos2, info);
								if (info.element_index != -1)
								{
									uint32_t idx = depthIndexCompact(info.depth, x);
									atomicMin(hitIndex + l, idx);
									// printf("intersect pos: %f, %f, lbm pos: %d %d %d\n", info.hit_point[0], info.hit_point[1], i, j, l);
								}
							}
						}
					}
				}
			}
		}

		__global__ void mrDeformableSolver2DKernel_Shrd(
			float *fMom, float *fMomPost, LbmNodeFlag *flags, uint32_t *hitIndex, float *fluidForce,
			SolidGeometryProxy_Device<2> solid_proxy,
			int sample_x, int sample_y,
			float vis_shear,
			float cl, float ct, float cf)
		{
			extern __shared__ half s_pop[];

			const int BLOCK_NX = blockDim.x;
			const int BLOCK_NY = blockDim.y;
			const int SHAREDMEM_PITCH = (BLOCK_NX + 2) * (BLOCK_NY + 2);

			LbmUtilsFuncGpu2D mrutilfunc;

			int currentId = threadIdx.x + BLOCK_NX * threadIdx.y;

			while (currentId < SHAREDMEM_PITCH)
			{
				int smIdx = (currentId) % (BLOCK_NX + 2);
				int smIdy = (currentId) / (BLOCK_NX + 2);

				int globalIdx = min(sample_x - 1, max(0, (smIdx - 1) + blockDim.x * blockIdx.x));
				int globalIdy = min(sample_y - 1, max(0, (smIdy - 1) + blockDim.y * blockIdx.y));

				int cellId = globalIdy * sample_x + globalIdx;

				float pop[9]; // = { 0 };
				mrutilfunc.CalculateDistributionD2Q9ALL(
					fMom[6 * cellId + 0],
					fMom[6 * cellId + 1],
					fMom[6 * cellId + 2],
					fMom[6 * cellId + 3],
					fMom[6 * cellId + 5],
					fMom[6 * cellId + 4],
					pop);

				int smId = smIdx + smIdy * (BLOCK_NX + 2);
				s_pop[0 * SHAREDMEM_PITCH + smId] = pop[0];
				s_pop[1 * SHAREDMEM_PITCH + smId] = pop[1];
				s_pop[2 * SHAREDMEM_PITCH + smId] = pop[2];
				s_pop[3 * SHAREDMEM_PITCH + smId] = pop[3];
				s_pop[4 * SHAREDMEM_PITCH + smId] = pop[4];
				s_pop[5 * SHAREDMEM_PITCH + smId] = pop[5];
				s_pop[6 * SHAREDMEM_PITCH + smId] = pop[6];
				s_pop[7 * SHAREDMEM_PITCH + smId] = pop[7];
				s_pop[8 * SHAREDMEM_PITCH + smId] = pop[8];
				s_pop[9 * SHAREDMEM_PITCH + smId] = pop[9];
				currentId += BLOCK_NX * BLOCK_NY;
			}

			__syncthreads();

			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;
			int curind = y * sample_x + x;

			if (
				(x >= 0 && x <= sample_x - 1) &&
				(y >= 0 && y <= sample_y - 1))
			{

				if (flags[curind] == LbmNodeFlag::Fluid || flags[curind] == LbmNodeFlag::SolidDynamic) // || (mlflow[0].flag[curind] >= ML_OUTLET_LEFT && mlflow[0].flag[curind] <= ML_OUTLET_UP))
				{
					float pop[9];

					float ux_cur = fMom[curind * 6 + 1];
					float uy_cur = fMom[curind * 6 + 2];
					float pixx_cur = fMom[curind * 6 + 3];
					float piyy_cur = fMom[curind * 6 + 4];
					float pixy_cur = fMom[curind * 6 + 5];

					for (int i = 0; i < 9; i++)
					{
						int dx = LbmD2Q9::c_ex[i];
						int dy = LbmD2Q9::c_ey[i];

						int x1 = x - dx;
						int y1 = y - dy;
						bool notCross = x1 >= 0 && x1 < sample_x && y1 >= 0 && y1 < sample_y;
						x1 = (x1 + sample_x) % sample_x;
						y1 = (y1 + sample_y) % sample_y;
						// z1 = (z - dz + sample_z) % sample_z;

						int ind_back = y1 * sample_x + x1;

						bool is_blocked = false;

						// Bounce back
						if (flags[ind_back] == LbmNodeFlag::Solid ||
							flags[ind_back] == LbmNodeFlag::WallLeft ||
							flags[ind_back] == LbmNodeFlag::WallRight ||
							flags[ind_back] == LbmNodeFlag::WallUp ||
							flags[ind_back] == LbmNodeFlag::WallDown ||
							flags[ind_back] == LbmNodeFlag::Wall)
						{
							int smIdx = threadIdx.x + 1;
							int smIdy = threadIdx.y + 1;
							pop[i] = max(0.0f, s_pop[LbmD2Q9::c_inv[i] * SHAREDMEM_PITCH + smIdx + smIdy * (BLOCK_NX + 2)]);
						}

						else if (notCross && hitIndex[9 * curind + i] != (uint32_t)-1)
						{
							IntersectionInfo info;
							float ux = 0, uy = 0;
							info.element_index = hitIndex[9 * curind + i] & 0x0000FFFF;
							solid_proxy.intersectAtIndex(glm::vec2(x * cl, y * cl), glm::vec2(x1 * cl, y1 * cl), info);
							is_blocked = info.element_index != -1;

							if (is_blocked)
							{
								float rhoVar = fMom[curind * 6 + 0];
								// printf("blocked %d %d %d\n", curind, info.element_index, hitIndex[9*curind+i]);

								glm::vec2 v;
								solid_proxy.getVelocityAt(info, v);
								ux = v[0];
								uy = v[1];
								ux *= ct / cl;
								uy *= ct / cl;
								float pixx = pixx_cur + ux * ux - ux_cur * ux_cur;
								float piyy = piyy_cur + uy * uy - uy_cur * uy_cur;
								float pixy = pixy_cur + uy * ux - ux_cur * uy_cur;
								float fin = 0, fout = 0;
								int smIdx = threadIdx.x + 1;
								int smIdy = threadIdx.y + 1;
								fin = max(0.0f, s_pop[LbmD2Q9::c_inv[i] * SHAREDMEM_PITCH + smIdx + smIdy * (BLOCK_NX + 2)]);
								mrutilfunc.CalculateDistributionD2Q9AtIndex(rhoVar, ux, uy, pixx, pixy, piyy, i, fout);

								pop[i] = fout;
								float fx = (fin * (-dx - ux) - fout * (dx - ux)) * cf;
								float fy = (fin * (-dy - uy) - fout * (dy - uy)) * cf;
								solid_proxy.addForceAt(info, glm::vec2(fx, fy));
							}
						}

						if (!is_blocked)
						{
							int smIdx = threadIdx.x + 1 - dx;
							int smIdy = threadIdx.y + 1 - dy;
							pop[i] = max(0.0f, s_pop[i * SHAREDMEM_PITCH + smIdx + smIdy * (BLOCK_NX + 2)]);
						}
					}

					float rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8];
					float FX = 0; // mlflow[0].fluidForce[2 * curind] - (rhoVar * domain_ax) / ( uop * uop / (labma * labma * deltax));
					float FY = 0; // mlflow[0].fluidForce[2 * curind + 1] - (rhoVar * domain_ay) / ( uop * uop / (labma * labma * deltax));
					float invRho = 1 / rhoVar;
					float ux = ((pop[1] - pop[3] + pop[5] - pop[6] - pop[7] + pop[8]) + 0.5 * FX) * invRho;
					float uy = ((pop[2] - pop[4] + pop[5] + pop[6] - pop[7] - pop[8]) + 0.5 * FY) * invRho;
					float pixx = pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[8];
					float piyy = pop[2] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8];
					float pixy = pop[5] - pop[6] + pop[7] - pop[8];

					float Omega = 1 / ((vis_shear) * 3.0 + 0.5f);
					mrutilfunc.Collision(
						rhoVar,
						ux, uy, FX, FY,
						Omega,
						pixx, piyy, pixy);

					pixx = 1 * (pixx * invRho - 1.0 * LbmD2Q9::c_cs2);
					piyy = 1 * (piyy * invRho - 1.0 * LbmD2Q9::c_cs2);
					pixy = 1 * (pixy * invRho);

					fMomPost[curind * 6 + 0] = rhoVar;
					fMomPost[curind * 6 + 1] = ux + FX * invRho / 2.0f;
					fMomPost[curind * 6 + 2] = uy + FY * invRho / 2.0f;
					fMomPost[curind * 6 + 3] = pixx;
					fMomPost[curind * 6 + 4] = piyy;
					fMomPost[curind * 6 + 5] = pixy;
				}
			}
		}

		__global__ void mrApplyOutlet2DKernel(
			float *fMom, LbmNodeFlag *flags, int sample_x, int sample_y, int sample_num)
		{
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;
			int curind = y * sample_x + x;
			if (
				(x >= 0 && x <= sample_x - 1) &&
				(y >= 0 && y <= sample_y - 1))
			{

				if (flags[curind] == LbmNodeFlag::OutletUp)
				{
					int indp = (y - 1) * sample_x + x;
					if (flags[indp] == LbmNodeFlag::Fluid)
					{
						float up = fMom[indp * 6 + 2];
						if (up > 0)
							fMom[curind * 6 + 2] = up;
						fMom[curind * 6 + 4] = up * up;
					}
				}

				else if (flags[curind] == LbmNodeFlag::OutletDown)
				{
					int indp = (y + 1) * sample_x + x;
					if (flags[indp] == LbmNodeFlag::Fluid)
					{
						float up = fMom[indp * 6 + 2];
						if (up < 0)
							fMom[curind * 6 + 2] = up;
						fMom[curind * 6 + 4] = up * up;
					}
				}

				else if (flags[curind] == LbmNodeFlag::OutletLeft)
				{
					int indp = y * sample_x + x + 1;
					if (flags[indp] == LbmNodeFlag::Fluid)
					{
						float up = fMom[indp * 6 + 1];
						if (up < 0)
							fMom[curind * 6 + 1] = up;
						fMom[curind * 6 + 3] = up * up;
					}
				}

				else if (flags[curind] == LbmNodeFlag::OutletRight)
				{
					int indp = y * sample_x + x - 1;
					if (flags[indp] == LbmNodeFlag::Fluid)
					{
						float up = fMom[indp * 6 + 1];
						if (up > 0)
							fMom[curind * 6 + 1] = up;
						fMom[curind * 6 + 3] = up * up;
					}
				}
			}
		}

		__global__ void resetSolidFlagKernel(
			LbmNodeFlag *flags, int sample_x, int sample_y)
		{
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;
			int curind = y * sample_x + x;
			if (
				(x >= 0 && x <= sample_x - 1) &&
				(y >= 0 && y <= sample_y - 1))
			{
				if (flags[curind] == LbmNodeFlag::Fluid || flags[curind] == LbmNodeFlag::SolidDynamic)
				{
					flags[curind] = LbmNodeFlag::Fluid;
				}
			}
		}

		__global__ void updateSolidFlagKernel(
			LbmNodeFlag *flags, SolidGeometryProxy_Device<2> solid_proxy, int sample_x, int sample_y, float delta_x)
		{
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;
			int curind = y * sample_x + x;
			if (
				(x >= 0 && x <= sample_x - 1) &&
				(y >= 0 && y <= sample_y - 1))
			{
				if (flags[curind] == LbmNodeFlag::Fluid)
				{
					vec2_t p = vec2_t((x + 0.5f) * delta_x, (y + 0.5f) * delta_x);
					if (solid_proxy.isInside(p))
					{
						flags[curind] = LbmNodeFlag::SolidDynamic;
					}
				}
			}
		}

		namespace fsi_coupling
		{
			void solveLbmAndFsiStep2D(
				LbmFlowField2D &flow_field,
				const SolidGeometryProxy_Device<2> &solid_proxy,
				const SimulationParameters2D &params,
				cudaStream_t stream)
			{
				// --- 1. preparation ---
				const int sample_x = flow_field.getNx();
				const int sample_y = flow_field.getNy();
				const int sample_num = sample_x * sample_y;
				const float vis_shear = flow_field.m_viscosity;
				const float cl = params.fluid_dx;
				const float ct = params.dt;
				const float cf = params.getCf();
				const int ecnt = solid_proxy.num_elements;

				// --- 2. precompute hit index ---

				CUDA_CHECK(cudaMemsetAsync(flow_field.m_hitIndex.data(), 0xFF, sample_num * 9 * sizeof(uint32_t), stream));

				const int occ_block_size = KernelConfig::BLOCK_SIZE_1D; // 使用定义的常量, e.g., 512
				const int occ_grid_size = (ecnt * 8 * 9 + occ_block_size - 1) / occ_block_size;

				mrSolver2DComputeOccupancyFlagKernel_v3<<<occ_grid_size, occ_block_size, 0, stream>>>(
					flow_field.m_hitIndex.data(),
					solid_proxy,
					sample_x, sample_y,
					params.fluid_dx,
					ecnt);

				CUDA_CHECK_KERNEL();
				// CUDA_CHECK(cudaStreamSynchronize(stream));

				// --- 3. core FSI calculation ---

				const dim3 block(
					KernelConfig::BLOCK_SIZE_2D_X,
					KernelConfig::BLOCK_SIZE_2D_Y);
				const dim3 grid(
					(sample_x + block.x - 1) / block.x,
					(sample_y + block.y - 1) / block.y);
				const size_t shared_mem_bytes = 9 *
												(block.x + 2) *
												(block.y + 2) *
												sizeof(half);

				mrDeformableSolver2DKernel_Shrd<<<grid, block, shared_mem_bytes, stream>>>(
					flow_field.m_current_moments->data(),
					flow_field.m_next_moments->data(),
					flow_field.m_flags.data(),
					flow_field.m_hitIndex.data(),
					flow_field.m_fluidForce.data(),
					solid_proxy,
					sample_x, sample_y,
					vis_shear,
					cl, ct, cf);
				CUDA_CHECK_KERNEL();
				// CUDA_CHECK(cudaStreamSynchronize(stream));

				// --- 4. Host-side swap ---

				flow_field.swapMoments();

				// --- 5. apply outlet boundary condition ---

				mrApplyOutlet2DKernel<<<grid, block, 0, stream>>>(
					flow_field.m_current_moments->data(),
					flow_field.m_flags.data(),
					sample_x, sample_y,
					sample_num);
				CUDA_CHECK_KERNEL();
				// CUDA_CHECK(cudaStreamSynchronize(stream));

				CUDA_CHECK(cudaStreamSynchronize(stream));
			}

			void fillSolidFlags2D(
				LbmFlowField2D &flow_field,
				const SolidGeometryProxy_Device<2> &solid_proxy,
				const SimulationParameters2D &params,
				cudaStream_t stream)
			{
				const int sample_x = flow_field.getNx();
				const int sample_y = flow_field.getNy();
				const float cl = params.fluid_dx;

				const dim3 block(
					KernelConfig::BLOCK_SIZE_2D_X,
					KernelConfig::BLOCK_SIZE_2D_Y);
				const dim3 grid(
					(sample_x + block.x - 1) / block.x,
					(sample_y + block.y - 1) / block.y);

				resetSolidFlagKernel<<<grid, block, 0, stream>>>(
					flow_field.m_flags.data(),
					sample_x, sample_y);
				CUDA_CHECK_KERNEL();

				updateSolidFlagKernel<<<grid, block, 0, stream>>>(
					flow_field.m_flags.data(),
					solid_proxy,
					sample_x, sample_y,
					cl);
				CUDA_CHECK_KERNEL();

				CUDA_CHECK(cudaStreamSynchronize(stream));
			}
		}

	} // namespace lbm
} // namespace fsi