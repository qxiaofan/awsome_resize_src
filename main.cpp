
#include <iostream>
 
#include<time.h>
 
#include <opencv2/opencv.hpp>
 
using namespace cv;
using namespace std;
 
typedef unsigned char uchar;
 
const int INTER_RESIZE_COEF_BITS = 11;
const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
 
const int INTER_REMAP_COEF_BITS = 15;
static const int MAX_ESIZE = 16;
const int INTER_REMAP_COEF_SCALE = 1 << INTER_REMAP_COEF_BITS;
typedef void(*ResizeFunc)(const Mat& src, Mat& dst,
	const int* xofs, const void* alpha,
	const int* yofs, const void* beta,
	int xmin, int xmax, int ksize);
 
 
 
typedef uchar value_type;
typedef int buf_type;
typedef short alpha_type;
 
typedef uchar T;
typedef int WT;
typedef short AT;
 
 
//typedef uchar value_type;
//typedef int buf_type;
//typedef short alpha_type;
 
//template<typename T, typename WT, typename AT>;
static inline int clip(int x, int a, int b)
{
	return x >= a ? (x < b ? x : b - 1) : a;
}
 
 
void hresize(const uchar** src, int** dst, int count,
	const int* xofs, const short* alpha,
	int swidth, int dwidth, int cn, int xmin, int xmax)
{
	int ONE = 2048;
	int dx, k;
	//VecOp vecOp;
 
	int dx0 = 0;//vecOp((const uchar**)src, (uchar**)dst, count, xofs, (const uchar*)alpha, swidth, dwidth, cn, xmin, xmax);
 
	for (k = 0; k <= count - 2; k++)
	{
		const T *S0 = src[k], *S1 = src[k + 1];
		WT *D0 = dst[k], *D1 = dst[k + 1];
		for (dx = dx0; dx < xmax; dx++)
		{
			int sx = xofs[dx];
			WT a0 = alpha[dx * 2], a1 = alpha[dx * 2 + 1];
			WT t0 = S0[sx] * a0 + S0[sx + cn] * a1;
			WT t1 = S1[sx] * a0 + S1[sx + cn] * a1;
			D0[dx] = t0; D1[dx] = t1;
		}
 
		for (; dx < dwidth; dx++)
		{
			int sx = xofs[dx];
			D0[dx] = WT(S0[sx] * ONE); D1[dx] = WT(S1[sx] * ONE);
		}
	}
 
	for (; k < count; k++)
	{
		const T *S = src[k];
		WT *D = dst[k];
		for (dx = 0; dx < xmax; dx++)
		{
			int sx = xofs[dx];
			D[dx] = S[sx] * alpha[dx * 2] + S[sx + cn] * alpha[dx * 2 + 1];
		}
 
		for (; dx < dwidth; dx++)
			D[dx] = WT(S[xofs[dx]] * ONE);
	}
}
 
 
 
void vresize(const buf_type** src, value_type* dst, const alpha_type* beta, int width)
{
	alpha_type b0 = beta[0], b1 = beta[1];
	const buf_type *S0 = src[0], *S1 = src[1];
	//VResizeLinearVec_32s8u vecOp;
 
	int x = 0;//vecOp((const uchar**)src, (uchar*)dst, (const uchar*)beta, width);
#if CV_ENABLE_UNROLLED
	for (; x <= width - 4; x += 4)
	{
		dst[x + 0] = uchar((((b0 * (S0[x + 0] >> 4)) >> 16) + ((b1 * (S1[x + 0] >> 4)) >> 16) + 2) >> 2);
		dst[x + 1] = uchar((((b0 * (S0[x + 1] >> 4)) >> 16) + ((b1 * (S1[x + 1] >> 4)) >> 16) + 2) >> 2);
		dst[x + 2] = uchar((((b0 * (S0[x + 2] >> 4)) >> 16) + ((b1 * (S1[x + 2] >> 4)) >> 16) + 2) >> 2);
		dst[x + 3] = uchar((((b0 * (S0[x + 3] >> 4)) >> 16) + ((b1 * (S1[x + 3] >> 4)) >> 16) + 2) >> 2);
	}
#endif
	for (; x < width; x++)
		dst[x] = uchar((((b0 * (S0[x] >> 4)) >> 16) + ((b1 * (S1[x] >> 4)) >> 16) + 2) >> 2);
 
 
}
 
void invoker(const Mat& src, Mat &dst, const int *xofs, const int *yofs,
	const alpha_type* alpha, const alpha_type* _beta, const Size& ssize, const Size &dsize,
	int ksize, int xmin, int xmax, Range& range)
{
	int dy, cn = src.channels();
	//HResize hresize;
	//VResize vresize;
 
	int bufstep = (int)alignSize(dsize.width, 16);
	AutoBuffer<buf_type> _buffer(bufstep*ksize);
	const value_type* srows[MAX_ESIZE] = { 0 };
	buf_type* rows[MAX_ESIZE] = { 0 };
	int prev_sy[MAX_ESIZE];
 
	for (int k = 0; k < ksize; k++)
	{
		prev_sy[k] = -1;
		rows[k] = (buf_type*)_buffer + bufstep*k;
	}
 
	const alpha_type* beta = _beta + ksize * range.start;
 
	for (dy = range.start; dy < range.end; dy++, beta += ksize)
	{
		int sy0 = yofs[dy], k0 = ksize, k1 = 0, ksize2 = ksize / 2;
 
		for (int k = 0; k < ksize; k++)
		{
			int sy = clip(sy0 - ksize2 + 1 + k, 0, ssize.height);
			for (k1 = std::max(k1, k); k1 < ksize; k1++)
			{
				if (sy == prev_sy[k1]) // if the sy-th row has been computed already, reuse it.
				{
					if (k1 > k)
						memcpy(rows[k], rows[k1], bufstep*sizeof(rows[0][0]));
					break;
				}
			}
			if (k1 == ksize)
				k0 = std::min(k0, k); // remember the first row that needs to be computed
			srows[k] = (value_type*)(src.data + src.step*sy);
			prev_sy[k] = sy;
		}
		
		if (k0 < ksize)
			hresize((const value_type**)(srows + k0), (buf_type**)(rows + k0), ksize - k0, xofs, (const alpha_type*)(alpha),
			ssize.width, dsize.width, cn, xmin, xmax);
		vresize((const buf_type**)rows, (value_type*)(dst.data + dst.step*dy), beta, dsize.width);

	}
 
}
 
static void resizeGeneric_(const Mat& src, Mat& dst,
	const int* xofs, const void* _alpha,
	const int* yofs, const void* _beta,
	int xmin, int xmax, int ksize)
{
	typedef alpha_type AT;
 
	const AT* beta = (const AT*)_beta;
	Size ssize = src.size(), dsize = dst.size();
	int cn = src.channels();
	ssize.width *= cn;
	dsize.width *= cn;
	xmin *= cn;
	xmax *= cn;
	// image resize is a separable operation. In case of not too strong
 
	Range range(0, dsize.height);
	//resizeGeneric_Invoker<HResize, VResize> invoker(src, dst, xofs, yofs, (const AT*)_alpha, beta,
	//ssize, dsize, ksize, xmin, xmax);
	//parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
	invoker(src, dst, xofs, yofs, (const AT*)_alpha, beta,
		ssize, dsize, ksize, xmin, xmax, range);
}
void Resize___(InputArray _src, OutputArray _dst, Size dsize,
	double inv_scale_x, double inv_scale_y, int interpolation)
{
	Mat src = _src.getMat();
	Size ssize = src.size();
 
	CV_Assert(ssize.area() > 0);
	CV_Assert(dsize.area() || (inv_scale_x > 0 && inv_scale_y > 0));
	if (!dsize.area())
	{
		dsize = Size(saturate_cast<int>(src.cols*inv_scale_x),
			saturate_cast<int>(src.rows*inv_scale_y));
		CV_Assert(dsize.area());
	}
	else
	{
		inv_scale_x = (double)dsize.width / src.cols;
		inv_scale_y = (double)dsize.height / src.rows;
	}
	_dst.create(dsize, src.type());
	Mat dst = _dst.getMat();
 
	int depth = src.depth(), cn = src.channels();
	double scale_x = 1. / inv_scale_x, scale_y = 1. / inv_scale_y;
	int k, sx, sy, dx, dy;
 
 
	{
		int iscale_x = saturate_cast<int>(scale_x);
		int iscale_y = saturate_cast<int>(scale_y);
 
		bool is_area_fast = std::abs(scale_x - iscale_x) < DBL_EPSILON &&
			std::abs(scale_y - iscale_y) < DBL_EPSILON;
 
		// in case of scale_x && scale_y is equal to 2
		// INTER_AREA (fast) also is equal to INTER_LINEAR
		if (interpolation == INTER_LINEAR && is_area_fast && iscale_x == 2 && iscale_y == 2)
		{
			interpolation = INTER_AREA;
		}
 
	}
 
	int xmin = 0, xmax = dsize.width, width = dsize.width*cn;
	bool area_mode = interpolation == INTER_AREA;
	bool fixpt = depth == CV_8U;
	float fx, fy;
	//ResizeFunc func = 0;
	int ksize = 0, ksize2;
	ksize = 2;
	//func = linear_tab[depth];
	ksize2 = ksize / 2;
 
	//CV_Assert(func != 0);
 
	AutoBuffer<uchar> _buffer((width + dsize.height)*(sizeof(int)+sizeof(float)*ksize));
	int* xofs = (int*)(uchar*)_buffer;
	int* yofs = xofs + width;
	float* alpha = (float*)(yofs + dsize.height);
	short* ialpha = (short*)alpha;
	float* beta = alpha + width*ksize;
	short* ibeta = ialpha + width*ksize;
	float cbuf[MAX_ESIZE];
 
	for (dx = 0; dx < dsize.width; dx++)
	{
		if (!area_mode)
		{
			fx = (float)((dx + 0.5)*scale_x - 0.5);
			sx = cvFloor(fx);
			fx -= sx;
		}
		else
		{
			sx = cvFloor(dx*scale_x);
			fx = (float)((dx + 1) - (sx + 1)*inv_scale_x);
			fx = fx <= 0 ? 0.f : fx - cvFloor(fx);
		}
 
		if (sx < ksize2 - 1)
		{
			xmin = dx + 1;
			if (sx < 0)
				fx = 0, sx = 0;
		}
 
		if (sx + ksize2 >= ssize.width)
		{
			xmax = std::min(xmax, dx);
			if (sx >= ssize.width - 1)
				fx = 0, sx = ssize.width - 1;
		}
 
		for (k = 0, sx *= cn; k < cn; k++)
			xofs[dx*cn + k] = sx + k;
 
		cbuf[0] = 1.f - fx;
		cbuf[1] = fx;
 
		if (fixpt)
		{
			for (k = 0; k < ksize; k++)
				ialpha[dx*cn*ksize + k] = saturate_cast<short>(cbuf[k] * INTER_RESIZE_COEF_SCALE);
			for (; k < cn*ksize; k++)
				ialpha[dx*cn*ksize + k] = ialpha[dx*cn*ksize + k - ksize];
		}
		else
		{
			for (k = 0; k < ksize; k++)
				alpha[dx*cn*ksize + k] = cbuf[k];
			for (; k < cn*ksize; k++)
				alpha[dx*cn*ksize + k] = alpha[dx*cn*ksize + k - ksize];
		}
	}
 
	for (dy = 0; dy < dsize.height; dy++)
	{
		if (!area_mode)
		{
			fy = (float)((dy + 0.5)*scale_y - 0.5);
			sy = cvFloor(fy);
			fy -= sy;
		}
		else
		{
			sy = cvFloor(dy*scale_y);
			fy = (float)((dy + 1) - (sy + 1)*inv_scale_y);
			fy = fy <= 0 ? 0.f : fy - cvFloor(fy);
		}
 
		yofs[dy] = sy;
		cbuf[0] = 1.f - fy;
		cbuf[1] = fy;
 
 
		if (fixpt)
		{
			for (k = 0; k < ksize; k++)
				ibeta[dy*ksize + k] = saturate_cast<short>(cbuf[k] * INTER_RESIZE_COEF_SCALE);
		}
		else
		{
			for (k = 0; k < ksize; k++)
				beta[dy*ksize + k] = cbuf[k]; 
		}
	}
 
	resizeGeneric_(src, dst, xofs, fixpt ? (void*)ialpha : (void*)alpha, yofs,
		fixpt ? (void*)ibeta : (void*)beta, xmin, xmax, ksize);
}
 
 
 
int main()
{
	string strfile = "../src.jpg";
	Mat src = cv::imread(strfile);
	int imageWidth = src.cols;
	int imageHeight = src.rows;
	std::cout<<"src.cols == "<<imageWidth<<std::endl;
    std::cout<<"src.rows == "<<imageHeight<<std::endl;
	cvtColor(src, src, CV_BGR2GRAY);
	int dst_width = 600;
	int dst_height = 400;
	Size dsize(dst_width, dst_height);
	Mat dst;
	clock_t start, finish;
	double totaltime;
	start = clock();
 
	int interpolation = INTER_LINEAR;//双线性插值（默认方法）
	Resize___(src, dst, dsize, 0, 0, interpolation);
 
	finish = clock();
	totaltime = (double)(finish - start);
	cout << "\n此程序1的运行时间为" << totaltime << "毫秒！" << endl;
 
	imwrite("../resize_img.jpg", dst);
 
return 1;
 
}

 
 
 


 

 
 

