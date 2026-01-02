// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/common/CudaCommon.cuh"

#include "FSI_Simulator/core/FsiCoupling.hpp"
#include "FSI_Simulator/core/SolidGeometryProxy_Device.cuh"

#include "FSI_Simulator/lbm/LbmDataTypes.cuh"
#include "FSI_Simulator/lbm/LbmUtilsFunc.cuh"
#include "FSI_Simulator/lbm/LbmConstants.hpp"

namespace fsi
{

	namespace lbm
	{
		namespace LbmD3Q27
		{
			extern __constant__ float c_w[Q];
			extern __constant__ float c_ex[Q];
			extern __constant__ float c_ey[Q];
			extern __constant__ float c_ez[Q];
			extern __constant__ float c_cs2;
			extern __constant__ int c_inv[Q];
		}

		__global__ void mrSolver3DComputeOccupancyFlagKernel_v3(uint32_t *hit_index, SolidGeometryProxy_Device<3> solid_proxy, int sample_x, int sample_y, int sample_z, float dx, int ecnt)
		{
			int idx = threadIdx.x + blockDim.x * blockIdx.x;
			int xid = idx / 26;
			int l = idx % 26 + 1;
			int x = xid / 27;
			int id = xid % 27;
			int x_id = id / 9;
			int y_id = (id % 9) / 3;
			int z_id = id % 3;
			if (x >= 0 && x < ecnt)
			{
				glm::vec3 min, max;
				solid_proxy.getBoundingBox(x, min, max);
				min /= dx;
				max /= dx;

				for (int i = int(min[0]) - 1 + x_id; i <= int(max[0]) + 1; i += 3)
				{
					if (i >= 0 && i < sample_x)
					{
						for (int j = int(min[1]) - 1 + y_id; j <= int(max[1]) + 1; j += 3)
						{
							if (j >= 0 && j < sample_y)
							{
								for (int k = int(min[2]) - 1 + z_id; k <= int(max[2]) + 1; k += 3)
								{
									if (k >= 0 && k < sample_z)
									{
										int ind = k * sample_y * sample_x + j * sample_x + i;
										uint32_t *hitIndex = hit_index + ind * 27;
										IntersectionInfo info;

										glm::vec3 pos1(i, j, k), pos2(i - LbmD3Q27::c_ex[l], j - LbmD3Q27::c_ey[l], k - LbmD3Q27::c_ez[l]);
										pos1 *= dx;
										pos2 *= dx;
										info.element_index = x;
										solid_proxy.intersectAtIndex(pos1, pos2, info);
										if (info.element_index != -1)
										{
											uint32_t idx = depthIndexCompact(info.depth, x);
											atomicMin(hitIndex + l, idx);
										}
									}
								}
							}
						}
					}
				}
			}
		}

		__global__ void mrDeformableSolver3DKernel_ShrdM(
			float *fMom, float *fMomPost, LbmNodeFlag *flags, uint32_t *hitIndex, float *force,
			SolidGeometryProxy_Device<3> solid_proxy,
			int sample_x, int sample_y, int sample_z,
			int sample_num, float vis_shear,
			float cl, float ct, float cf)
		{
			const int BLOCK_NX = blockDim.x;
			const int BLOCK_NY = blockDim.y;
			const int BLOCK_NZ = blockDim.z;

			__shared__ float s_pop[27][KernelConfig::SHAREDMEM_LBM_SIZE_3D];

			LbmUtilsFuncGpu3D mrutilfunc;

			int currentId = threadIdx.x + BLOCK_NX * threadIdx.y + BLOCK_NX * BLOCK_NY * threadIdx.z;

			while (currentId < KernelConfig::SHAREDMEM_LBM_SIZE_3D)
			{
				int smIdx = (currentId % ((BLOCK_NX + 2) * (BLOCK_NY + 2))) % (BLOCK_NX + 2);
				int smIdy = (currentId % ((BLOCK_NX + 2) * (BLOCK_NY + 2))) / (BLOCK_NX + 2);
				int smIdz = currentId / ((BLOCK_NX + 2) * (BLOCK_NY + 2));

				int globalIdx = min(sample_x - 1, max(0, (smIdx - 1) + blockDim.x * blockIdx.x));
				int globalIdy = min(sample_y - 1, max(0, (smIdy - 1) + blockDim.y * blockIdx.y));
				int globalIdz = min(sample_z - 1, max(0, (smIdz - 1) + blockDim.z * blockIdx.z));

				int cellId = globalIdz * sample_x * sample_y + globalIdy * sample_x + globalIdx;

				float pop[27]; // = { 0 };
				mrutilfunc.CalculateDistributionD3Q27All(
					fMom[10 * cellId + 0],
					fMom[10 * cellId + 1],
					fMom[10 * cellId + 2],
					fMom[10 * cellId + 3],
					fMom[10 * cellId + 4],
					fMom[10 * cellId + 5],
					fMom[10 * cellId + 6],
					fMom[10 * cellId + 7],
					fMom[10 * cellId + 8],
					fMom[10 * cellId + 9],
					pop);

				int smId = smIdx + smIdy * (BLOCK_NX + 2) + smIdz * (BLOCK_NX + 2) * (BLOCK_NY + 2);
				s_pop[0][smId] = pop[0];
				s_pop[1][smId] = pop[1];
				s_pop[2][smId] = pop[2];
				s_pop[3][smId] = pop[3];
				s_pop[4][smId] = pop[4];
				s_pop[5][smId] = pop[5];
				s_pop[6][smId] = pop[6];
				s_pop[7][smId] = pop[7];
				s_pop[8][smId] = pop[8];
				s_pop[9][smId] = pop[9];
				s_pop[10][smId] = pop[10];
				s_pop[11][smId] = pop[11];
				s_pop[12][smId] = pop[12];
				s_pop[13][smId] = pop[13];
				s_pop[14][smId] = pop[14];
				s_pop[15][smId] = pop[15];
				s_pop[16][smId] = pop[16];
				s_pop[17][smId] = pop[17];
				s_pop[18][smId] = pop[18];
				s_pop[19][smId] = pop[19];
				s_pop[20][smId] = pop[20];
				s_pop[21][smId] = pop[21];
				s_pop[22][smId] = pop[22];
				s_pop[23][smId] = pop[23];
				s_pop[24][smId] = pop[24];
				s_pop[25][smId] = pop[25];
				s_pop[26][smId] = pop[26];

				// if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
				// {
				// 	printf("cell_id: %d, smId: %d\n\
		// 	fMom: %f %f %f %f %f %f %f %f %f %f\n\
		// 	pop:  %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
				// 	cellId, smId,
				// 	fMom[10 * cellId + 0], fMom[10 * cellId + 1], fMom[10 * cellId + 2], fMom[10 * cellId + 3], fMom[10 * cellId + 4], fMom[10 * cellId + 5], fMom[10 * cellId + 6], fMom[10 * cellId + 7], fMom[10 * cellId + 8], fMom[10 * cellId + 9],
				// 	pop[0], pop[1], pop[2], pop[3], pop[4], pop[5], pop[6], pop[7], pop[8], pop[9], pop[10], pop[11], pop[12], pop[13], pop[14], pop[15], pop[16], pop[17], pop[18], pop[19], pop[20], pop[21], pop[22], pop[23], pop[24], pop[25], pop[26]);
				// }

				currentId += BLOCK_NX * BLOCK_NY * BLOCK_NZ;
			}

			__syncthreads();

			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;
			int z = threadIdx.z + blockDim.z * blockIdx.z;
			int curind = z * sample_y * sample_x + y * sample_x + x;
			if (
				(x >= 0 && x <= sample_x - 1) &&
				(y >= 0 && y <= sample_y - 1) &&
				(z >= 0 && z <= sample_z - 1))
			{
				if (flags[curind] == LbmNodeFlag::Fluid || flags[curind] == LbmNodeFlag::SolidDynamic)
				{
					float pop[27];

					float rhoVar_cur = fMom[curind * 10 + 0];
					float ux_cur = fMom[curind * 10 + 1];
					float uy_cur = fMom[curind * 10 + 2];
					float uz_cur = fMom[curind * 10 + 3];
					float pixx_cur = fMom[curind * 10 + 4];
					float piyy_cur = fMom[curind * 10 + 5];
					float pizz_cur = fMom[curind * 10 + 6];
					float pixy_cur = fMom[curind * 10 + 7];
					float piyz_cur = fMom[curind * 10 + 8];
					float pixz_cur = fMom[curind * 10 + 9];

					for (int i = 0; i < 27; i++)
					{
						int dx = int(LbmD3Q27::c_ex[i]);
						int dy = int(LbmD3Q27::c_ey[i]);
						int dz = int(LbmD3Q27::c_ez[i]);
						int x1 = x - dx;
						int y1 = y - dy;
						int z1 = z - dz;
						bool notCross = x1 >= 0 && x1 < sample_x && y1 >= 0 && y1 < sample_y && z1 >= 0 && z1 < sample_z;
						x1 = (x1 + sample_x) % sample_x;
						y1 = (y1 + sample_y) % sample_y;
						z1 = (z1 + sample_z) % sample_z;

						int ind_back = z1 * sample_y * sample_x + y1 * sample_x + x1;
						// Bounce back
						if (flags[ind_back] == LbmNodeFlag::Solid ||
							flags[ind_back] == LbmNodeFlag::Wall ||
							flags[ind_back] == LbmNodeFlag::WallRight ||
							flags[ind_back] == LbmNodeFlag::WallUp ||
							flags[ind_back] == LbmNodeFlag::WallDown ||
							flags[ind_back] == LbmNodeFlag::WallFront ||
							flags[ind_back] == LbmNodeFlag::WallBack ||
							flags[ind_back] == LbmNodeFlag::WallLeft)
						{
							int smIdx = threadIdx.x + 1;
							int smIdy = threadIdx.y + 1;
							int smIdz = threadIdx.z + 1;
							pop[i] = max(0.0f, s_pop[LbmD3Q27::c_inv[i]][smIdx + smIdy * (BLOCK_NX + 2) + smIdz * (BLOCK_NX + 2) * (BLOCK_NY + 2)]);
						}

						else
						{
							bool intersect = notCross && hitIndex[27 * curind + i] != (uint32_t)-1;
							if (intersect)
							{
								glm::vec3 pos1(x, y, z), pos2(x1, y1, z1), u;
								pos1 *= cl;
								pos2 *= cl;
								IntersectionInfo info;
								info.element_index = hitIndex[27 * curind + i] & 0x0000FFFF;
								solid_proxy.intersectAtIndex(pos1, pos2, info);

								if (intersect && info.element_index != -1)
								{
									// auto hit_inv = hitIndex[27*ind_back+LbmD3Q27::c_inv[i]] & 0x0000FFFF;
									// if (info.element_index != hit_inv)
									// {
									// 	printf("Warning: element index mismatch! %d vs %d\n", info.element_index, hit_inv);
									// }
									solid_proxy.getVelocityAt(info, u);
									u = u * ct / cl;
									float ux = u[0];
									float uy = u[1];
									float uz = u[2];
									float pixx = pixx_cur + ux * ux - ux_cur * ux_cur;
									float piyy = piyy_cur + uy * uy - uy_cur * uy_cur;
									float pizz = pizz_cur + uz * uz - uz_cur * uz_cur;
									float pixy = pixy_cur + ux * uy - ux_cur * uy_cur;
									float piyz = piyz_cur + uy * uz - uy_cur * uz_cur;
									float pixz = pixz_cur + ux * uz - ux_cur * uz_cur;

									mrutilfunc.CalculateDistributionD3Q27AtIndex(
										rhoVar_cur, ux, uy, uz, pixx, piyy, pizz, pixy, piyz, pixz, i, pop[i]);

									int smIdx = threadIdx.x + 1;
									int smIdy = threadIdx.y + 1;
									int smIdz = threadIdx.z + 1;
									float fin = max(0.0f, s_pop[LbmD3Q27::c_inv[i]][smIdx + smIdy * (BLOCK_NX + 2) + smIdz * (BLOCK_NX + 2) * (BLOCK_NY + 2)]);

									float fout = pop[i];
									float fx = (fin * (-dx - ux) - fout * (dx - ux)) * cf;
									float fy = (fin * (-dy - uy) - fout * (dy - uy)) * cf;
									float fz = (fin * (-dz - uz) - fout * (dz - uz)) * cf;

									glm::vec3 F(fx, fy, fz);
									solid_proxy.addForceAt(info, F);
									// printf("id: %d %d %d,  link: %d, fin: %f, fout: %f, px: %f, py: %f, pz: %f, cf: %f, fx: %f, fy: %f, fz: %f\n",
									// x, y, z, i, fin, fout, info.pHit[0], info.pHit[1], info.pHit[2], cf, fx, fy, fz);
								}
								else
									intersect = false;
							}

							if (!intersect)
							{
								int smIdx = threadIdx.x + 1 - dx;
								int smIdy = threadIdx.y + 1 - dy;
								int smIdz = threadIdx.z + 1 - dz;
								pop[i] = max(0.0f, s_pop[i][smIdx + smIdy * (BLOCK_NX + 2) + smIdz * (BLOCK_NX + 2) * (BLOCK_NY + 2)]);
							}
						}
					}

					float FX = force[3 * curind + 0];
					float FY = force[3 * curind + 1];
					float FZ = force[3 * curind + 2];

					float rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
					float invRho = 1 / rhoVar;
					float ux = ((pop[1] + pop[7] + pop[9] + pop[13] + pop[15] + pop[19] + pop[21] + pop[23] + pop[26]) - (pop[2] + pop[8] + pop[10] + pop[14] + pop[16] + pop[20] + pop[22] + pop[24] + pop[25]) + 0.5 * FX) * invRho;
					float uy = ((pop[3] + pop[7] + pop[11] + pop[14] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25]) - (pop[4] + pop[8] + pop[12] + pop[13] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26]) + 0.5 * FY) * invRho;
					float uz = ((pop[5] + pop[9] + pop[11] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25]) - (pop[6] + pop[10] + pop[12] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26]) + 0.5 * FZ) * invRho;
					float pixx = ((pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]));
					float piyy = ((pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]));
					float pizz = ((pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]));
					float pixy = (((pop[7] + pop[8] + pop[19] + pop[20] + pop[21] + pop[22]) - (pop[13] + pop[14] + pop[23] + pop[24] + pop[25] + pop[26])));
					float piyz = (((pop[11] + pop[12] + pop[19] + pop[20] + pop[25] + pop[26]) - (pop[17] + pop[18] + pop[21] + pop[22] + pop[23] + pop[24])));
					float pixz = (((pop[9] + pop[10] + pop[19] + pop[20] + pop[23] + pop[24]) - (pop[15] + pop[16] + pop[21] + pop[22] + pop[25] + pop[26])));

					float Omega = 1 / ((vis_shear) * 3.0 + 0.5f);
					mrutilfunc.Collision(
						rhoVar,
						ux, uy, uz, FX, FY, FZ,
						Omega,
						pixx, piyy, pizz, pixy, piyz, pixz);

					fMomPost[curind * 10 + 0] = rhoVar;
					fMomPost[curind * 10 + 1] = ux + FX * invRho / 2.0f;
					fMomPost[curind * 10 + 2] = uy + FY * invRho / 2.0f;
					fMomPost[curind * 10 + 3] = uz + FZ * invRho / 2.0f;

					fMomPost[curind * 10 + 4] = pixx * invRho - LbmD3Q27::c_cs2;
					fMomPost[curind * 10 + 5] = piyy * invRho - LbmD3Q27::c_cs2;
					fMomPost[curind * 10 + 6] = pizz * invRho - LbmD3Q27::c_cs2;
					fMomPost[curind * 10 + 7] = pixy * invRho;
					fMomPost[curind * 10 + 8] = piyz * invRho;
					fMomPost[curind * 10 + 9] = pixz * invRho;

					// if (x == 5 && y == 50 && z == 50)
					// {
					// 	printf("f(5, 50, 50):");
					// 	for (int i = 0; i < 27; i++)
					// 	{
					// 		printf("%f ", s_pop[i][3 + 3 * (BLOCK_NX + 2) + 3 * (BLOCK_NX + 2) * (BLOCK_NY + 2)]);
					// 	}
					// 	printf("\n");
					// 	// printf("m(5, 50, 50): %f %f %f %f %f %f %f %f %f %f\n", fMom[curind * 10 + 0], fMom[curind * 10 + 1], fMom[curind * 10 + 2], fMom[curind * 10 + 3], fMom[curind * 10 + 4], fMom[curind * 10 + 5], fMom[curind * 10 + 6], fMom[curind * 10 + 7], fMom[curind * 10 + 8], fMom[curind * 10 + 9]);
					// }
				}
			}
		}

		__global__ void mrApplyOutlet3DKernel(
			float *fMom, LbmNodeFlag *flags, int sample_x, int sample_y, int sample_z, int sample_num)
		{
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;
			int z = threadIdx.z + blockDim.z * blockIdx.z;
			int curind = z * sample_y * sample_x + y * sample_x + x;
			if (
				(x >= 0 && x <= sample_x - 1) &&
				(y >= 0 && y <= sample_y - 1) &&
				(z >= 0 && z <= sample_z - 1))
			{
				if (flags[curind] == LbmNodeFlag::OutletUp)
				{
					int indp = (z - 1) * sample_y * sample_x + y * sample_x + x;
					if (flags[indp] == LbmNodeFlag::Fluid)
					{
						float up = fMom[indp * 10 + 3];
						if (up > 0)
							fMom[curind * 10 + 3] = up;
						fMom[curind * 10 + 6] = up * up;
					}
				}

				else if (flags[curind] == LbmNodeFlag::OutletDown)
				{
					int indp = (z + 1) * sample_y * sample_x + y * sample_x + x;
					if (flags[indp] == LbmNodeFlag::Fluid)
					{
						float up = fMom[indp * 10 + 3];
						if (up < 0)
							fMom[curind * 10 + 3] = up;
						fMom[curind * 10 + 6] = up * up;
					}
				}

				else if (flags[curind] == LbmNodeFlag::OutletLeft)
				{
					int indp = z * sample_y * sample_x + y * sample_x + x + 1;
					if (flags[indp] == LbmNodeFlag::Fluid)
					{
						float up = fMom[indp * 10 + 1];
						if (up < 0)
							fMom[curind * 10 + 1] = up;
						fMom[curind * 10 + 4] = up * up;
					}
				}

				else if (flags[curind] == LbmNodeFlag::OutletRight)
				{
					int indp = z * sample_y * sample_x + y * sample_x + x - 1;
					if (flags[indp] == LbmNodeFlag::Fluid)
					{
						float up = fMom[indp * 10 + 1];
						if (up > 0)
							fMom[curind * 10 + 1] = up;
						fMom[curind * 10 + 4] = up * up;
					}
				}

				else if (flags[curind] == LbmNodeFlag::OutletFront)
				{
					int indp = z * sample_y * sample_x + (y + 1) * sample_x + x;
					if (flags[indp] == LbmNodeFlag::Fluid)
					{
						float up = fMom[indp * 10 + 2];
						if (up < 0)
							fMom[curind * 10 + 2] = up;
						fMom[curind * 10 + 5] = up * up;
					}
				}

				else if (flags[curind] == LbmNodeFlag::OutletBack)
				{
					int indp = z * sample_y * sample_x + (y - 1) * sample_x + x;
					if (flags[indp] == LbmNodeFlag::Fluid)
					{
						float up = fMom[indp * 10 + 2];
						if (up > 0)
							fMom[curind * 10 + 2] = up;
						fMom[curind * 10 + 5] = up * up;
					}
				}
			}
		}

		namespace fsi_coupling
		{
			void solveLbmAndFsiStep3D(
				LbmFlowField3D &flow_field,
				const SolidGeometryProxy_Device<3> &solid_proxy,
				const SimulationParameters3D &params,
				cudaStream_t stream)
			{
				const int sample_x = flow_field.getNx();
				const int sample_y = flow_field.getNy();
				const int sample_z = flow_field.getNz();
				const int sample_num = sample_x * sample_y * sample_z;
				const float vis_shear = flow_field.getViscosity();
				const float cl = params.fluid_dx;
				const float ct = params.dt;
				const float cf = params.getCf();
				const int ecnt = solid_proxy.num_elements;

				CUDA_CHECK(cudaMemsetAsync(flow_field.m_hitIndex.data(), 0xFF, sample_num * 27 * sizeof(uint32_t), stream));

				const int occ_block_size = KernelConfig::BLOCK_SIZE_1D; // 使用定义的常量, e.g., 512
				const int occ_grid_size = (ecnt * 26 * 27 + occ_block_size - 1) / occ_block_size;

				mrSolver3DComputeOccupancyFlagKernel_v3<<<occ_grid_size, occ_block_size, 0, stream>>>(
					flow_field.m_hitIndex.data(),
					solid_proxy,
					sample_x, sample_y, sample_z,
					params.fluid_dx,
					ecnt);

				CUDA_CHECK_KERNEL();

				const dim3 block(
					KernelConfig::BLOCK_SIZE_3D_X,
					KernelConfig::BLOCK_SIZE_3D_Y,
					KernelConfig::BLOCK_SIZE_3D_Z);
				const dim3 grid(
					(sample_x + block.x - 1) / block.x,
					(sample_y + block.y - 1) / block.y,
					(sample_z + block.z - 1) / block.z);

				mrDeformableSolver3DKernel_ShrdM<<<grid, block, 0, stream>>>(
					flow_field.m_current_moments->data(),
					flow_field.m_next_moments->data(),
					flow_field.m_flags.data(),
					flow_field.m_hitIndex.data(),
					flow_field.m_fluidForce.data(),
					solid_proxy,
					sample_x, sample_y, sample_z,
					sample_num,
					vis_shear,
					cl, ct, cf);
				CUDA_CHECK_KERNEL();

				flow_field.swapMoments();

				mrApplyOutlet3DKernel<<<grid, block, 0, stream>>>(
					flow_field.m_current_moments->data(),
					flow_field.m_flags.data(),
					sample_x, sample_y, sample_z,
					sample_num);
				CUDA_CHECK_KERNEL();
			}
		}

	} // namespace lbm
} // namespace fsi