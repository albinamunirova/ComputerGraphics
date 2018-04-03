#include "align.h"
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cmath>

using std::string;
using std::cout;
using std::endl;

void beg_detection(int i, int j, int *beg_x1, int *beg_y1, int *beg_x2, int *beg_y2)
{
	if (i < 0 && j < 0)
			{
				*beg_x1 = 0;
				*beg_x2 = -i;
				*beg_y1 = 0;
				*beg_y2 = -j;
			}
			else 
					if (i >= 0 && j < 0)
					{
						*beg_x1 = i;
						*beg_x2 = 0;
						*beg_y1 = 0;
						*beg_y2 = -j;
					}
					else
							if (i >= 0 && j >= 0)
							{
								*beg_x1 = i;
								*beg_x2 = 0;
								*beg_y1 = j;
								*beg_y2 = 0;
							}
							else
									if (i < 0 && j >= 0)
									{
										*beg_x1 = 0;
										*beg_x2 = -i;
										*beg_y1 = j;
										*beg_y2 = 0;
									}
}

void beg_detection_pyramid(int i, int j, int *beg_x1, int *beg_y1, int *beg_x2, int *beg_y2, int x, int y)
{
	if (i + x < 0 && j + y < 0)
			{
				*beg_x1 = 0;
				*beg_x2 = -i - x;
				*beg_y1 = 0;
				*beg_y2 = -j - y;
			}
			else 
					if (i + x >= 0 && j + y < 0)
					{
						*beg_x1 = i + x;
						*beg_x2 = 0;
						*beg_y1 = 0;
						*beg_y2 = -j - y;
					}
					else
							if (i + x >= 0 && j + y >= 0)
							{
								*beg_x1 = i + x;
								*beg_x2 = 0;
								*beg_y1 = j + y;
								*beg_y2 = 0;
							}
							else
									if (i +x < 0 && j + y >= 0)
									{
										*beg_x1 = 0;
										*beg_x2 = -i - x;
										*beg_y1 = j + y;
										*beg_y2 = 0;
									}
}

void pixel_checking(int *r1, int *g1, int *b1)
{
		if (*r1 < 0) 
		{
			*r1 = 0;
		}
		else
				if (*r1 > 255) 
				{
					*r1 = 255;
				}

		if (*g1 < 0) 
		{
			*g1 = 0;
		}
		else
				if (*g1 > 255) 
				{
					*g1 = 255;
				}

		if (*b1 < 0) 
		{
			*b1 = 0;
		}
		else
				if (*b1 > 255) 
				{
					*b1 = 255;
				}
}

void ideal_offset(Image im1, Image im2, int *x, int *y)
{
	int i,j = 0;
	int min_metr = 255*255*255;
	int mse = 0;
	int ideal_x = 0;
	int ideal_y = 0;
	Image im1_cut = im1;
	Image im2_cut = im2;
	int beg_x1 = 0, beg_x2 = 0, beg_y1 = 0, beg_y2 = 0;
	for(i = -4; i < 5; i++)
	{
		for (j = -2; j < 3; j++)
		{
			beg_detection_pyramid(i,j,&beg_x1, &beg_y1, &beg_x2, &beg_y2, *x, *y);
			im1_cut = im1.submatrix(beg_x1,beg_y1,im1.n_rows-abs(i + *x),im1.n_cols-abs(j + *y));
			im2_cut = im2.submatrix(beg_x2,beg_y2,im2.n_rows-abs(i + *x),im2.n_cols-abs(j + *y));
			uint r1,g1,b1,r2,g2,b2;
			for (uint k = 0.29*im1_cut.n_rows; k < 0.78*im1_cut.n_rows; k++)
			{
				for (uint m = 0.29*im1_cut.n_cols; m < 0.78*im1_cut.n_cols; m++)
				{
					std:: tie(r1,g1,b1) = im1_cut(k,m);
					std:: tie(r2,g2,b2) = im2_cut(k,m);
					mse+= (r1-r2)*(r1-r2);
				}
			}
			mse = mse /(im1_cut.n_rows*im1_cut.n_cols);
			if(mse < min_metr)
			{
				min_metr = mse;
				ideal_x  = i + *x;
				ideal_y  = j + *y;
			}
		}
	}
	*x = ideal_x;
	*y = ideal_y;
}


void qs(int *s_arr, int first, int last)
{
    if (first < last)
    {
        int left = first, right = last, middle = s_arr[(left + right) / 2];
        do
        {
            while (s_arr[left] < middle) left++;
            while (s_arr[right] > middle) right--;
            if (left <= right)
            {
                int tmp = s_arr[left];
                s_arr[left] = s_arr[right];
                s_arr[right] = tmp;
                left++;
                right--;
            }
        } while (left <= right);
        qs(s_arr, first, right);
        qs(s_arr, left, last);
    }
}

Image align(Image srcImage, bool isPostprocessing, std::string postprocessingType, double fraction, bool isMirror, 
            bool isInterp, bool isSubpixel, double subScale)
{
	if (isSubpixel == 1) 
	{
		srcImage = resize(srcImage, subScale);
	}
	uint height = srcImage.n_rows/3;
	uint width = srcImage.n_cols;

	int offset_x1 = 0, offset_x2 = 0, offset_y1 = 0, offset_y2 = 0;

	int count = round(log(3*height/300)/log(2));
	double scale = pow(2,-count);

	for (int i = 0; i < count+1; i++)
	{
			Image scaleImage = resize(srcImage,scale);
			Image RedImage = scaleImage.submatrix(2*height*scale,0,height*scale,width*scale);
			Image BlueImage = scaleImage.submatrix(0,0,height*scale,width*scale);
			Image GreenImage = scaleImage.submatrix(height*scale,0,height*scale,width*scale);

			scale*=2;
			offset_x1*=2;
			offset_y1*=2;
			offset_x2*=2;
			offset_y2*=2;

			ideal_offset(GreenImage,RedImage,&offset_x1,&offset_y1);
			ideal_offset(GreenImage,BlueImage,&offset_x2,&offset_y2);
	}

	Image RedImage = srcImage.submatrix(2*height,0,height,width);
	Image BlueImage = srcImage.submatrix(0,0,height,width);
	Image GreenImage = srcImage.submatrix(height,0,height,width);



	Image img = GreenImage.deep_copy();

	int beg_x1 = 0, beg_x2 = 0, beg_y1 = 0, beg_y2 = 0;

	beg_detection(offset_x1,offset_y1,&beg_x1, &beg_y1, &beg_x2, &beg_y2);
	for (uint i = beg_x1; i < img.n_rows-abs(offset_x1); i++)
	{
		for (uint j = beg_y1; j < img.n_cols-abs(offset_y1); j++)
		{
				std::get<0>(img(i,j)) = std::get<0>(RedImage(i-offset_x1,j-offset_y1));
		}
	}

	beg_detection(offset_x2,offset_y2,&beg_x1, &beg_y1, &beg_x2, &beg_y2);
	for (uint i = beg_x1; i < img.n_rows-abs(offset_x2); i++)
	{
		for (uint j = beg_y1; j < img.n_cols-abs(offset_y2); j++)
		{
				std::get<2>(img(i,j)) = std::get<2>(BlueImage(i-offset_x2,j-offset_y2));
		}
	}
	if (isSubpixel == 1) 
	{
		img = resize(img, 1/subScale);
	}
	if(postprocessingType == "--gray-world")
	{
		img = gray_world(img);
	}

	if(postprocessingType == "--autocontrast")
	{
		img = autocontrast(img,fraction);
	}

	if(postprocessingType == "--unsharp")
	{
		img = unsharp(img);
	}
	return img;
}

Image sobel_x(Image src_image) {
    Matrix<double> kernel = {{-1, 0, 1},
                             {-2, 0, 2},
                             {-1, 0, 1}};
    return custom(src_image, kernel);
}

Image sobel_y(Image src_image) {
    Matrix<double> kernel = {{ 1,  2,  1},
                             { 0,  0,  0},
                             {-1, -2, -1}};
    return custom(src_image, kernel);
}

Image mirror(Image src_image, uint radius)
{
	int r,g,b;
	Image img(src_image.n_rows + 2*radius, src_image.n_cols + 2*radius);

	for(uint i = 0; i < src_image.n_rows; i++)
	{
		for (uint j = 0; j < src_image.n_cols; j++)
		{
			std:: tie(r,g,b) = src_image(i,j);
			img(i + radius,j + radius) = std::make_tuple(r,g,b);
		}
	}

	for(uint i = 0; i < radius; i++)
	{
		for (uint j = 0; j < src_image.n_cols; j++)
		{			
			std:: tie(r,g,b) = src_image(i,j);
			img(radius - i - 1,radius + j)=std::make_tuple(r,g,b);
		}
	}

	for(uint i = src_image.n_rows - radius; i < src_image.n_rows; i++)
	{
		for (uint j = 0; j < src_image.n_cols; j++)
		{			
			std:: tie(r,g,b) = src_image(i,j);
			img(radius + 2*src_image.n_rows - i - 1,j + radius)=std::make_tuple(r,g,b);
		}
	}

	Image img1 = img.deep_copy();
	for(uint i = 0; i < src_image.n_rows + 2*radius; i++)
	{
		for (uint j = 0; j < radius; j++)
		{			
			std:: tie(r,g,b) = img1(i,j);
			img(i,radius - j - 1)=std::make_tuple(r,g,b);
		}
	}

	for(uint i = 0; i < src_image.n_rows + 2*radius; i++)
	{
		for (uint j = src_image.n_cols - radius; j < src_image.n_cols; j++)
		{			
			std:: tie(r,g,b) = img1(i,j);
			img(i,radius + 2*src_image.n_cols -j - 1)=std::make_tuple(r,g,b);
		}
	}
			return img;
}

Image unsharp(Image src_image) 
{
		mirror(src_image,1);
		int r,g,b;
    Matrix<double> kernel = {{-0.1666, -0.6666, -0.1666},
                       			 {-0.6666, 4.3333, -0.6666},
                        		 {-0.1666, -0.6666, -0.1666}};

		for(uint i = 1; i < src_image.n_rows-1; i++)
		{
			for (uint j = 1; j < src_image.n_cols-1; j++)
			{
				std:: tie(r,g,b) = src_image(i,j);
				int r_sum = 0, g_sum = 0, b_sum = 0;
				for (int k = -1; k < 2; k++)
				{
					for (int m = -1; m < 2; m++)
					{
						int r1,g1,b1;
						std:: tie(r1,g1,b1) = src_image(i+k,j+m);
						r_sum += kernel(k+1,m+1)*r1;
						g_sum += kernel(k+1,m+1)*g1;
						b_sum += kernel(k+1,m+1)*b1;
					}
				}
				pixel_checking(&r_sum,&g_sum,&b_sum);
			  src_image(i,j)=std::make_tuple(r_sum,g_sum,b_sum);
			}
		}
    return src_image;
}

Image gray_world(Image src_image) 
{
		int r_sum = 0,g_sum = 0,b_sum = 0,r1,g1,b1;
		for(uint i = 0; i < src_image.n_rows; i++)
		{
			for (uint j = 0; j < src_image.n_cols; j++)
			{
				std:: tie(r1,g1,b1) = src_image(i,j);
				r_sum+=r1;
				g_sum+=g1;
				b_sum+=b1;
			}
		}
		r_sum = r_sum/(src_image.n_rows*src_image.n_cols);
		g_sum = g_sum/(src_image.n_rows*src_image.n_cols);
		b_sum = b_sum/(src_image.n_rows*src_image.n_cols);
		double sum = (r_sum + g_sum + b_sum)/3;
		double r = sum/r_sum;
		double g = sum/g_sum;
		double b = sum/b_sum;
		for(uint i = 0; i < src_image.n_rows; i++)
		{
			for (uint j = 0; j < src_image.n_cols; j++)
			{
				std:: tie(r1,g1,b1) = src_image(i,j);
				r1*= r;
				g1*= g;
				b1*= b;
				pixel_checking(&r1,&g1,&b1);
			src_image(i,j)=std::make_tuple(r1,g1,b1);
		 }
	 }
   return src_image;
}

Image resize(Image src_image, double scale) 
{
		int oldh = src_image.n_rows;
		int oldw = src_image.n_cols;
		int newh = src_image.n_rows*scale;
		int neww = src_image.n_cols*scale;
		Image new_image(newh,neww);
		int i, j;
		int h, w;
		float t;
		float u;
		float tmp;
		float d1, d2, d3, d4;
		int r1, r2, r3, r4;	
		int g1, g2, g3, g4;
		int b1, b2, b3, b4;

		int red, green, blue;

		for (j = 0; j < newh; j++) {
			tmp = j * (oldh - 1) / (newh - 1) ;
			h = tmp;
			if (h < 0) {
				h = 0;
			} else {
				if (h >= oldh - 1) {
					h = oldh - 2;
				}
			}
			u = tmp - h;

			for (i = 0; i < neww; i++) {

				tmp = i * (oldw - 1)/ (neww - 1) ;
				w = tmp;
				if (w < 0) {
					w = 0;
				} else {
					if (w >= oldw - 1) {
						w = oldw - 2;
					}
				}
				t = tmp - w;

				d1 = (1 - t) * (1 - u);
				d2 = t * (1 - u);
				d3 = t * u;
				d4 = (1 - t) * u;
				std:: tie(r1,g1,b1) = src_image(h,w);
				std:: tie(r2,g2,b2) = src_image(h,w + 1);
				std:: tie(r3,g3,b3) = src_image(h + 1,w + 1);
				std:: tie(r4,g4,b4) = src_image(h + 1,w);
				blue = b1 *d1 + b2 *d2 + b3 *d3 + b4 *d4;
				green = g1 * d1 + g2 * d2 + g3 * d3 + g4 * d4;
				red = r1 * d1 + r2 * d2 + r3 * d3 + r4 * d4;
				pixel_checking(&red,&green,&blue);
			  new_image(j,i)=std::make_tuple(red,green,blue);
			}
		}
    return new_image;
}

Image custom(Image src_image, Matrix<double> kernel) {
    // Function custom is useful for making concrete linear filtrations
    // like gaussian or sobel. So, we assume that you implement custom
    // and then implement other filtrations using this function.
    // sobel_x and sobel_y are given as an example.
    return src_image;
}

Image autocontrast(Image src_image, double fraction) 
{
		int r,g,b;
		int gisto[256] = {0};
		int bright = 0;
		int count = 0;
		int min = 0, max = 0;
		int size_m = src_image.n_rows*src_image.n_cols*fraction;
		Matrix<int>pix (src_image.n_rows, src_image.n_cols);

		for(uint i = 0; i < src_image.n_rows; i++)
		{
			for (uint j = 0; j < src_image.n_cols; j++)
			{
				std:: tie(r,g,b) = src_image(i,j);
				bright = 0.2125*r + 0.7154*g + 0.0721*b;
				gisto[bright]++;
				pix(i,j) = bright;
			}
		}

		while (min <= size_m)
		{
			min+=gisto[count];
			count++;
		}

		min = count - 1;
		count = 255;

		while (max <= size_m)
		{
			max+=gisto[count];
			count--;
		}

		max = count + 1;

		for(uint i = 0; i < src_image.n_rows; i++)
		{
			for (uint j = 0; j < src_image.n_cols; j++)
			{
				std:: tie(r,g,b) = src_image(i,j);
				if (pix(i,j) < min)
				{
					r = 0;
					g = 0;
					b = 0;
				}
				else
					if (pix(i,j) > max)
					{
						r = 255;
						g = 255;
						b = 255;
					}
					else
					{
						r = (r - min)*255/(max-min);
						g = (g - min)*255/(max-min);
						b = (b - min)*255/(max-min);
						pixel_checking(&r,&g,&b);
					}
					src_image(i,j)=std::make_tuple(r,g,b);
			}
		}
    return src_image;
}


Image gaussian(Image src_image, double sigma, int radius)  {
    return src_image;
}

Image gaussian_separable(Image src_image, double sigma, int radius) {
    return src_image;
}

Image median(Image src_image, int radius) 
{	
		mirror(src_image,radius);
		int r,g,b;
		int h = src_image.n_rows;
		int w = src_image.n_cols;
		for(int i = radius; i < h-radius; i++)
		{
			for (int j = radius; j < w-radius; j++)
			{
				std:: tie(r,g,b) = src_image(i,j);
				int n = (2*radius+1)*(2*radius+1);
				int array_r[n];
				int array_g[n];
				int array_b[n];
				int s = 0;
				for (int k = -radius; k < radius +1 ; k++)
				{
					for (int m = -radius; m < radius +1; m++)
					{
						int r1,g1,b1;
						std:: tie(r1,g1,b1) = src_image(i+k,j+m);
						array_r[s] = r1;
						array_g[s] = g1;
						array_b[s] = b1;
						s++;
					}
				}
				qs(array_r,0,n-1);
				qs(array_g,0,n-1);
				qs(array_b,0,n-1);
				r = array_r[n/2];
				g = array_g[n/2];
				b = array_b[n/2];
			  src_image(i,j)=std::make_tuple(r,g,b);
			}
		}
    return src_image.submatrix(radius,radius,src_image.n_rows-2*radius,src_image.n_cols-2*radius);
}

int position(int a[], int n, int pixel)
{
	int m = 0;
	for (int i = 0; i < n; i++)
	{
		if(a[i] == pixel)
			m = i;
	}
	return m;
}

Image median_linear(Image src_image, int radius) 
{		
		mirror(src_image,radius);
		int r,g,b;
		int h = src_image.n_rows;
		int w = src_image.n_cols;
		Image tmp_image = src_image.deep_copy();
		int n = (2*radius+1)*(2*radius+1);
		int array_r[n];
		int array_g[n];
		int array_b[n];
		int r_pos,g_pos,b_pos;

		for(int i = radius; i < h-radius; i++)
		{
			for (int j = radius; j < w-radius; j++)
			{
				if (j == radius)
				{
					int s = 0;
					for (int k = -radius; k < radius +1 ; k++)
					{
						for (int m = -radius; m < radius +1; m++)
						{
							int r1,g1,b1;
							std:: tie(r1,g1,b1) = src_image(i+k,j+m);
							array_r[s] = r1;
							array_g[s] = g1;
							array_b[s] = b1;
							s++;
						}
					}
					qs(array_r,0,n-1);
					qs(array_g,0,n-1);
					qs(array_b,0,n-1);
					r = array_r[n/2];
					g = array_g[n/2];
					b = array_b[n/2];
				  tmp_image(i,j)=std::make_tuple(r,g,b);
				}
				else
				{
					for (int k = -radius; k < radius +1; k++)
					{
						int r1,g1,b1;
						std:: tie(r1,g1,b1) = src_image(i+k,j-radius-1);
						r_pos = position(array_r,n,r1);
						g_pos = position(array_g,n,g1);
						b_pos = position(array_b,n,b1);
						std:: tie(r1,g1,b1) = src_image(i+k,j+radius);
						array_r[r_pos] = r1;
						array_g[g_pos] = g1;
						array_b[b_pos] = b1;
					}
					qs(array_r,0,n-1);
					qs(array_g,0,n-1);
					qs(array_b,0,n-1);
					r = array_r[n/2];
					g = array_g[n/2];
					b = array_b[n/2];
				  tmp_image(i,j)=std::make_tuple(r,g,b);
				}
			}
		}
    return tmp_image.submatrix(radius,radius,src_image.n_rows-2*radius,src_image.n_cols-2*radius);
}

Image median_const(Image src_image, int radius) {
    return src_image;
}

Image canny(Image src_image, int threshold1, int threshold2) {
    return src_image;
}
