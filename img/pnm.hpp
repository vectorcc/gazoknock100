#include<iostream>
#include<string>
#include<string.h>
#include<stdio.h>

class PNM
{
	private:
		char mn[4];//magic number
		int w;
		int h;
		int bright;
		int chan;//1pxのbyte数
		int len;
	public:
		float **data;
		PNM(){data=NULL;}
		PNM(char *fname)
		{
			FILE *fp;
			if((fp=fopen(fname,"rb"))!=NULL)
			{
				fscanf(fp,"%s\n",this->mn);
				if(this->mn[0]=='P'&&(this->mn[1]=='5'||this->mn[1]=='6'))
					//とりあえずP5とp6だけ
				{
					this->chan=this->mn[1]=='5'?1:3;
					//P5ならグレースケールP6ならフルカラー
					fscanf(fp,"%d %d\n",&this->w,&this->h);
					fscanf(fp,"%d\n",&this->bright);
					int a;
					std::string s;
					while((a=fgetc(fp))!=EOF)
						s+=a;
					this->data=new float*[this->chan+4];
					for(int i=0;i<this->chan;++i)
						this->data[i]=new float[s.length()/this->chan+10];
					for(int i=0;i<s.length()/this->chan;++i)
						for(int j=0;j<this->chan;++j)
							this->data[j][i]=s[i*this->chan+j];
					this->len=s.length()/this->chan;
				}
			}
		}
		void h_dump(FILE *fp)
		{
			fprintf(fp,"%s\n%d %d\n%d\n",this->mn,this->w,this->h,this->bright);
		}

		bool set(char _mn[4],int _w,int _h,int _bright)
		{
			if(!(_mn[1]=='5'||_mn[1]=='6'))
				return false;
			strcpy(this->mn,_mn);
			this->w=_w;
			this->h=_h;
			this->bright=_bright;
			this->chan=_mn[1]=='5'?1:3;
			this->len=_w*_h/this->chan;
			this->data=new float*[this->chan+3];
			for(int i=0;i<this->chan;++i)
				this->data[i]=new float[this->len+10];
			return true;
		}

		PNM *gray()
		{
			if(this->chan!=3)
				return NULL;
			PNM *res=new PNM;
			res->set("P5",this->w,this->h,this->bright);
			res->h_dump(stdout);
			for(int i=0;i<this->len;++i)
				//>Y = 0.2126 R + 0.7152 G + 0.0722 B
				res->data[0][i]=0.2126*this->data[0][i]+0.7152*this->data[1][i]+0.0722*this->data[2][i];
				return res;
		}
		PNM *nichika(int t=127)
		{
			PNM *res;
			if((res=this->gray())==NULL)
				return NULL;
			printf("t:%d\n",t);
			unsigned char c;
			for(int i=0;c=res->data[0][i],i<res->len;++i)
				res->data[0][i]=c>t?255:0;
					
			return res;
		}

		void write(FILE *fp)
		{
			h_dump(fp);
			for(int i=0;i<this->len;++i)
				for(int j=0;j<this->chan;++j)
					fputc(((char)this->data[j][i]),fp);
		}
		PNM *otsu_nichika()
		{
			using namespace std;
			int t=0,t_,n1=0,n2;
			PNM *gray_;
			if((gray_=this->gray())==NULL)
				return NULL;
			float num0,num1;
			unsigned int sum0,sum1;
			unsigned int ave0,ave1;
			unsigned char c;
			for(t_=0;t_<256;t_+=1)
			{
			num0=0,num1=0;
			sum0=0,sum1=0;
			ave0=0,ave1=0;
			for(int i=0;c=this->data[0][i],i<this->len;++i)
				if(c>t_)
				{
					num0++;
					sum0+=c;
				}
				else
				{
					num1++;
					sum1+=c;
				}
			ave0=sum0/num0;
			ave1=sum1/num1;
			n2=num0/(num0+num1)*num1/(num0+num1)*(ave0-ave1)*(ave0-ave1);
			//printf("%d %f %f %f %d\n",n2,num0,num1,(ave0-ave1),t_);
			if(n1<n2)
			{
				t=t_;
				n1=n2;
			}
			}
			return this->nichika(t);
		}
};


