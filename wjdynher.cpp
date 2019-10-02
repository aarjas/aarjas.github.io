#include <RcppArmadillo.h>
#include <random>
#include <vector>
#include <cmath>
#include <time.h>
#include <stdexcept>
#include <omp.h>
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

const double pi = 3.141592653589793;
const double sigma = 1;
std::default_random_engine g;
std::uniform_real_distribution<> runif(0, 1);
std::normal_distribution<double> wgen(0, 1);
std::uniform_real_distribution<> thetagen(0, 2*pi);

// [[Rcpp::export]]
mat matern(const double& l, const int& N)
{
    mat C(N, N);
    for(int i = 0; i < N; i++)
    {
        for(int j = i; j < N; j++)
        {
            double dd = std::abs(i - j);
            if(dd > 0)
                C(i, j) = C(j, i) = (1 + sqrt(5)*dd/l + 5*pow(dd, 2)/(3*pow(l, 2)))*exp(-sqrt(5)*dd/l);
            else
                C(i, j) = 1;
        }
    }
    return C;
}

// [[Rcpp::export]]
mat sqec(const double& d, const double psi, const int& N)
{
    mat C(N, N);
    for(int i = 0; i < N; i++)
    {
        for(int j = i; j < N; j++)
        {
            double dd = std::abs(i - j)*d;
            C(i, j) = C(j, i) = std::exp(-0.5*((dd*dd)/(psi*psi)));
        }
    }
    return C;
}

double dnorm(const double x, const double mu, const double sd)
{
    double d = -0.5*std::log(2*pi) - std::log(sd) - 0.5*std::pow((x - mu)/sd, 2);
    return d;
}

// [[Rcpp::export]]
double inddnorm(const vec& x, const vec& mu, const double& sd)
{
    double d = 0;
    for(int i = 0; i < x.n_elem; i++)
    {
        d = d + dnorm(x(i), mu(i), sd);
    }
    return d;
}

double inddnorm2(const vec& x, const vec& mu, const vec& sd)
{
    double d = 0;
    for(int i = 0; i < x.n_elem; i++)
    {
        d = d + dnorm(x(i), mu(i), std::sqrt(std::exp(sd(i))));
    }
    return d;
}

double dmvnorm(const vec& x, const vec& mu, const mat& C)
{
    int N = x.n_elem;
    mat ex = (x - mu).t()*inv(C)*(x - mu);
    double y = ex(0, 0);
    double d = std::exp(-0.5*y)/(std::pow(2*pi, 0.5*N)*std::sqrt(det(C)));
    return std::log(d);
}

/*// [[Rcpp::export]]
double dmvnorm2(const vec& x, const vec& mu, const mat& Cinv, const double& logdetC)
{
    int N = x.n_elem;
    mat ex = (x - mu).t()*Cinv*(x - mu);
    double y = ex(0, 0);
    double d = -0.5*N*std::log(2*pi) - 0.5*logdetC - 0.5*y;
    return d;
}*/

// [[Rcpp::export]]
double dmvnorm2(const vec& x, const mat& Cinv, const double& logdetC)
{
    double y = as_scalar(x.t()*Cinv*x);
    return(-0.5*logdetC - 0.5*y);
}

/*
// [[Rcpp::export]]
double dmvnorm3(const vec& x, const vec& mu, const mat& Cinv, const double& logdetC)
{
    int N = x.n_elem;
    mat ex = (x - mu).t()*Cinv*(x - mu);
    double y = ex(0, 0);
    double d = -0.5*N*std::log(2*pi) + logdetC - 0.5*y;
    return d;
}
*/

// [[Rcpp::export]]
double dmvnorm3(const vec& x, const mat& Cinv, const double& logdetC)
{
    double y = as_scalar(x.t()*Cinv*x);
    return(logdetC - 0.5*y);
}


double dunif(const double x, const double min, const double max)
{
    if(x >= min && x <= max)
        return 1.0/(max - min);
    else
        return 0.0;
}



// [[Rcpp::export]]
mat L(const double& u, const double& d, const int& N)
{
    double eu = std::pow(u, 2);
    double d2 = std::pow(d, 2);
    mat Lu(N, N, fill::zeros);
    for(int i = 1; i < (N - 1); i++)
    {
        Lu(i, i) = 1 + 2*eu/d2;
        Lu(i, i + 1) = Lu(i, i - 1) = -eu/d2;
    }
    Lu(N - 1, N - 1) = Lu(0, 0) = 1 + 2*eu/d2;
    Lu(0, 1) = Lu(N - 1, N - 2) = Lu(0, N - 1) = Lu(N - 1, 0) = -eu/d2;
    return Lu/std::sqrt(4*u);
}

// [[Rcpp::export]]
mat Lns(vec u, const double& d, const int& N)
{
    u.transform([](double val){return std::exp(val);});
    //double d2 = std::pow(d, 2);
    double d2 = d*d;
    mat Lu(N, N, fill::zeros);
    for(int i = 1; i < (N - 1); i++)
    {
        double ui2 = u(i)*u(i);
        Lu(i, i) = (1 + 2*ui2/d2)/std::sqrt(4*u(i));
        Lu(i, i + 1) = Lu(i, i - 1) = (-ui2/d2)/std::sqrt(4*u(i));
    }
    Lu(N - 1, N - 1) = (1 + 2*std::pow(u(N - 1), 2)/d2)/std::sqrt(4*u(N - 1));
    Lu(0, 0) = (1 + 2*std::pow(u(0), 2)/d2)/std::sqrt(4*u(0));
    Lu(0, 1) = /*Lu(0, N - 1) =*/ (-std::pow(u(0), 2)/d2)/std::sqrt(4*u(0));
    /*Lu(N - 1, 0) = */Lu(N - 1, N - 2) = (-std::pow(u(N - 1), 2)/d2)/std::sqrt(4*u(N - 1));
    return Lu;
}

// [[Rcpp::export]]
mat L2(const double& u, const double& h, const int& N)
{
    mat Lu(N, N, fill::zeros);
    double a0 = (std::sqrt(h/u) + std::sqrt(h/u + 4*u/h))/std::sqrt(8);
    double a1 = (std::sqrt(h/u) - std::sqrt(h/u + 4*u/h))/std::sqrt(8);
    for(int i = 0; i < (N - 1); i++)
    {
        Lu(i, i) = a0;
        Lu(i, i + 1) = a1;
    }
    Lu(N - 1, N - 1) = 1;
    return Lu;
}

// [[Rcpp::export]]
rowvec Lnsrow(const double& u, const int& i, const int& N, const double& d)
{
    rowvec Lur(N, fill::zeros);
    //double d2 = std::pow(d, 2);
    double d2 = d*d;
    //double u2 = std::pow(u, 2);
    double u2 = u*u;
    if(i == 0)
    {
        Lur(0) = 1 + 2*u2/d2;
        Lur(1) = -u2/d2;
        //Lur(N - 1) = -u2/d2;
    }
    else if(i == (N - 1))
    {
        Lur(N - 2) = -u2/d2;
        Lur(N - 1) = 1 + 2*u2/d2;
        //Lur(0) = -u2/d2;
    }
    else
    {
        Lur(i - 1) = Lur(i + 1) = -u2/d2;
        Lur(i) = 1 + 2*u2/d2;
    }
    return Lur/std::sqrt(4*u);
}

// [[Rcpp::export]]
mat expc(const double& d, const double psi, const int& N)
{
    mat C(N, N);
    for(int i = 0; i < N; i++)
    {
        for(int j = i; j < N; j++)
        {
            double dd = std::abs(i - j)*d;
            C(i, j) = C(j, i) = sigma*std::exp(-dd/psi);
        }
    }
    return C;
}

// [[Rcpp::export]]
mat fastprod(const vec& v, mat m)
{
    for(int j = 0; j < m.n_cols; j++)
    {
        for(int i = 0; i < m.n_rows; i++)
        {
            m(i, j) = v(i)*v(j)*m(i, j);
        }
    }
    return m;
}

double digamma(const double& x, const double& alpha, const double& beta)
{
    if(x > 0)
        return(std::log((std::pow(beta, alpha)/std::tgamma(alpha))*std::pow(x, -alpha - 1)*std::exp(-beta/x)));
    else
        return 0;
}

// [[Rcpp::export]]
void ff()
{
    mat b(10, 10, fill::eye);
    b.diag().fill(1);
    mat a(10, 10, fill::randn);
    vec v(10, fill::ones);
    double c = 0;
    #pragma omp parallel for
    for(int i = 0; i < 10; i++)
    {
        c = c + dmvnorm2(v, b, 0);
    }
}

// [[Rcpp::export]]
mat dmmprod(const vec& a, mat b)
{
    #pragma omp parallel for
    for(int i = 0; i < b.n_cols; i++)
    {
        b.col(i) = a(i)*b.col(i);
    }
    return(b);
}

// [[Rcpp::export]]
double ldet(const mat& A)
{
	double val; double sign;
	log_det(sign, val, A);
	return(sign*val);
}

// [[Rcpp::export]]
void plot_r_cpp_call(const Rcpp::NumericVector& x){

  // Obtain environment containing function
  Rcpp::Environment graph("package:graphics"); 

  // Make function callable from C++
  Rcpp::Function plot_r = graph["plot"];  

  // Call the function and receive its list output
  plot_r(Rcpp::_["x"] = x, Rcpp::_["type"]  = "l", Rcpp::_["ylab"]  = "", Rcpp::_["xlab"]  = ""); // example of additional param

}

// [[Rcpp::export]]
void par_r_cpp_call(const vec& n){

  // Obtain environment containing function
  Rcpp::Environment gr("package:graphics"); 

  // Make function callable from C++
  Rcpp::Function par_r = gr["par"];  

  // Call the function and receive its list output
  par_r(Rcpp::_["mfrow"] = n); // example of additional param

}



// std calls the sd function in R
//[[Rcpp::export]]
double sd_sugar(const Rcpp::NumericVector& x){
  return Rcpp::sd(x); // uses Rcpp sugar
}

Rcpp::NumericVector arma2vec(const vec& x) {
    return Rcpp::NumericVector(x.begin(), x.end());
}


// [[Rcpp::export]]
Rcpp::List sim(const int& Nsim, mat Y, const mat& R, const int& cores, const double& ll)
{
    omp_set_num_threads(cores);
    const int M = R.n_cols; //yksilöiden määrä
    const int N = Y.n_cols;
    int mod = (Nsim + 0.0)/100;
    for(int i = 0; i < N; i++)
    {
       Y.col(i) = (Y.col(i) - mean(Y.col(i)));
    }
    double scale = sqrt(2)/mean(stddev(Y));
    Y = scale*Y;
    mat su(Nsim, N, fill::ones); su = su*std::log(1);
    mat se(Nsim, N, fill::ones); se = se*std::log(1);
	mat ser(Nsim, N, fill::zeros); mat sur(Nsim, N, fill::zeros);
    vec zer(M, fill::zeros);
    mat Id(M, M, fill::eye);
    int nz = 0;
    double ml = 0; double sl = 0.01;
    vec l(Nsim); l(0) = log(ll);
    mat Lz = matern(exp(l(0)), N);
	//Lz = L2(exp(l(0)), 1, N);
    mat LztLz = inv(Lz);
	double logdetLz = ldet(Lz);
	mat Uz; vec ksiz;
    eig_sym(ksiz, Uz, Lz);
    mat invLz = Uz*diagmat(sqrt(ksiz))*Uz.t();
    double mlu = 0; double slu = 0.01;
    vec lu(Nsim); lu(0) = log(ll);
    mat Lzu = matern(exp(lu(0)), N);
	//Lzu = L2(exp(lu(0)), 1, N);
    mat LzutLzu = inv(Lzu);
	double logdetLzu = ldet(Lzu);
    mat invLzu = Uz*diagmat(sqrt(ksiz))*Uz.t();
    eig_sym(ksiz, Uz, Lzu);
    mat U; vec ksi;
    eig_sym(ksi, U, R);
    mat Ut = U.t();
    mat dm(M, M, fill::eye);
    vec cse = invLz*se.row(0).t();
    vec csu = invLzu*su.row(0).t();
	vec aa(2);
	aa(0) = 2; aa(1) = 2;
	par_r_cpp_call(aa);
	double glike = 0;
	#pragma omp parallel for reduction(+:glike)
    for(int j = 0; j < N; j++)
    {
         double delta = std::exp(cse(j) - csu(j));
         vec lambda = ksi + delta;
         //dm.diag() = pow(lambda, -1);
         mat ic = (dmmprod(pow(lambda, - 1), U)*Ut)/std::exp(csu(j));
         double loglambda = sum(log(lambda));
         double logdetc = M*csu(j) + loglambda;
         glike = glike + dmvnorm2(Y.col(j), ic, logdetc);
    }
	//double likeet = dmvnorm3(se.row(0).t(), LztLz, logdetLz);
	//double likeut = dmvnorm3(su.row(0).t(), LzutLzu, logdetLzu);
	mat temppi(2, 2); temppi(0, 0) = temppi(1, 0) = 1; temppi(0, 1) = -1.96; temppi(1, 1) = 1.96;
	double aaa = log(1.0); double bbb = log(N - 1.0);
	vec ttt(2); ttt(0) = aaa; ttt(1) = bbb;
	vec temppi2 = inv(temppi)*ttt;
	double mul = temppi2(0); double taul = temppi2(1);
	temppi2.print();
	double lasu = 0; double lase = 0;
    for(int i = 1; i < Nsim; i++)
    {
        vec sup = randn(N);
        glike = glike + std::log(runif(g));
        double theta = thetagen(g);
        double thetamin = theta - 2*pi;
        double thetamax = theta;
        double like; vec csupp;
        do
        {
            vec supp = su.row(i - 1).t()*std::cos(theta) + sup*std::sin(theta);
            csupp = invLzu*supp;
            like = 0;
            #pragma omp parallel for reduction(+:like)
            for(int j = 0; j < N; j++)
            {
                double delta = std::exp(cse(j) - csupp(j));
                vec lambda = ksi + delta;
                //dm.diag() = pow(lambda, -1);
                mat icp = (dmmprod(pow(lambda, - 1), U)*Ut)/std::exp(csupp(j));
                double loglambda = sum(log(lambda));
                double logdetcp = M*csupp(j) + loglambda;
                like = like + dmvnorm2(Y.col(j), icp, logdetcp);
            }
            if(like > glike)
			{
                su.row(i) = supp.t();
			}
            else
            {
                if(theta < 0)
                    thetamin = theta;
                else
                    thetamax = theta;
                std::uniform_real_distribution<> thetagen2(thetamin, thetamax);
                theta = thetagen2(g);
            }
        }
        while(like <= glike);
		glike = like;
		csu = csupp;

        vec sep = randn(N);
        glike = glike + std::log(runif(g));
        double theta1 = thetagen(g);
        double thetamin1 = theta1 - 2*pi;
        double thetamax1 = theta1;
        double like1; vec csepp;
        do
        {
            vec sepp = se.row(i - 1).t()*std::cos(theta1) + sep*std::sin(theta1);
            csepp = invLz*sepp;
            like1 = 0;
            #pragma omp parallel for reduction(+:like1)
            for(int j = 0; j < N; j++)
            {
                double delta = std::exp(csepp(j) - csu(j));
                vec lambda = ksi + delta;
                //dm.diag() = pow(lambda, -1);
                mat icp = (dmmprod(pow(lambda, - 1), U)*Ut)/std::exp(csu(j));
                double loglambda = sum(log(lambda));
                double logdetcp = M*csu(j) + loglambda;
                like1 = like1 + dmvnorm2(Y.col(j), icp, logdetcp);
            }
            if(like1 > glike)
			{
                se.row(i) = sepp.t();
			}
            else
            {
                if(theta1 < 0)
                    thetamin1 = theta1;
                else
                    thetamax1 = theta1;
                std::uniform_real_distribution<> thetagen3(thetamin1, thetamax1);
                theta1 = thetagen3(g);
            }
        }
        while(like1 <= glike);
		glike = like1;
		cse = csepp;


        std::normal_distribution<double> lgen(l(i - 1), sqrt(sl));
        double lprop = lgen(g);
        mat Lzp = matern(exp(lprop), N);
		eig_sym(ksiz, Uz, Lzp);
        mat invLzp = Uz*diagmat(sqrt(ksiz))*Uz.t();
        vec csep = invLzp*se.row(i).t();
        double likee = 0;
        #pragma omp parallel for reduction(+:likee)
        for(int j = 0; j < N; j++)
        {
            double delta = std::exp(csep(j) - csu(j));
            vec lambda = ksi + delta;
            mat icp = (dmmprod(pow(lambda, - 1), U)*Ut)/std::exp(csu(j));
            double loglambda = sum(log(lambda));
            double logdetcp = M*csu(j) + loglambda;
            likee = likee + dmvnorm2(Y.col(j), icp, logdetcp);
        }
		double al = likee - glike + dnorm(lprop, mul, taul) - dnorm(l(i - 1), mul, taul);
        if(std::log(runif(g)) < al)
        {
            l(i) = lprop;
            invLz = invLzp;
            glike = likee;
            cse = csep;
            lase++;
        }
        else
            l(i) = l(i - 1);

		/*if(i % 50 == 0)
		{
		    nz++;
		    double delta = std::min(0.01, 1.0/sqrt(nz));
		    if(lase/i < 0.44)
                sl = sl - delta;
            else
                sl = sl + delta;

		}*/
        if(i > 50)
            sl = 2.38*2.38*var(l.subvec(0, i));

        std::normal_distribution<double> lugen(lu(i - 1), sqrt(slu));
        double luprop = lugen(g);
        mat Lzup = matern(exp(luprop), N);
		eig_sym(ksiz, Uz, Lzup);
        mat invLzup = Uz*diagmat(sqrt(ksiz))*Uz.t();
        vec csup = invLzup*su.row(i).t();
        double likeu = 0;
        #pragma omp parallel for reduction(+:likeu)
        for(int j = 0; j < N; j++)
        {
            double delta = std::exp(cse(j) - csup(j));
            vec lambda = ksi + delta;
            mat icp = (dmmprod(pow(lambda, - 1), U)*Ut)/std::exp(csup(j));
            double loglambda = sum(log(lambda));
            double logdetcp = M*csup(j) + loglambda;
            likeu = likeu + dmvnorm2(Y.col(j), icp, logdetcp);
        }
		double alu = likeu - glike + dnorm(luprop, mul, taul) - dnorm(lu(i - 1), mul, taul);
        if(std::log(runif(g)) < alu)
        {
            lu(i) = luprop;
            invLzu = invLzup;
            glike = likeu;
            csu = csup;
            lasu++;
        }
        else
            lu(i) = lu(i - 1);

		/*if(i % 50 == 0)
		{
		    double delta = std::min(0.01, 1.0/sqrt(nz));
		    if(lasu/i < 0.44)
                slu = slu - delta;
            else
                slu = slu + delta;

		}*/
        if(i > 50)
            slu = 2.38*2.38*var(lu.subvec(0, i));

		sur.row(i) = (invLzu*su.row(i).t()).t();
		ser.row(i) = (invLz*se.row(i).t()).t();




        if(i % mod == 0)
        {
            vec ab(1);
            ab(0) = (i + 0.0)/Nsim;
            ab.print();
        }
		
		if(i % 1000 == 0)
		{
			Rcpp::NumericVector x1 = arma2vec(ser.col(30).subvec(0, i));
			Rcpp::NumericVector x2 = arma2vec(sur.col(30).subvec(0, i));
			Rcpp::NumericVector x3 = arma2vec(l.subvec(0, i));
			Rcpp::NumericVector x4 = arma2vec(lu.subvec(0, i));
			plot_r_cpp_call(x1);
			plot_r_cpp_call(x2);
			plot_r_cpp_call(x3);
			plot_r_cpp_call(x4);
		}

    }
	
	


    return Rcpp::List::create(Rcpp::Named("su") = su,
                              Rcpp::Named("se") = se,
                              Rcpp::Named("l") = l,
                              Rcpp::Named("lu") = lu,
                              Rcpp::Named("scale") = scale,
							  Rcpp::Named("sur") = sur - 2*log(scale),
							  Rcpp::Named("ser") = ser - 2*log(scale));


}
