# example claim interarrival times
include("pTUM-base-5.0.jl")
using Distributions
using Random
using Plots

# parameters per category
cats_n=[25000 25000 25000]
cats_shape=[1.0 0.5 1.5]
cats_scale=[1.0 1.0 1.0]
cats_scale_update=[1.0 1.05 1.05]

Random.seed!(1234) # reproducable seed-value

data=zeros(0,6)
covers=zeros(0,2)
ncats=length(cats_n)

# prepare covers without claims
for catnr=1:length(cats_n)
  data=vcat(data,1.0*[catnr 0 2 1E18 1E18 0]) # prepare zero claims
end

# prepare claim observations
for catnr=1:ncats
  for i=1:cats_n[catnr]
    claims=0
    t0=0
    shape=cats_shape[catnr]
    scale=cats_scale[catnr]
    while t0<1
      #F_t0=1-exp(-(t0/scale)^shape)
      #t=cats_scale[catnr]*(-log((1-rand())*(1-F_t0)))^(1/cats_shape[catnr])-t0
      t=scale*(-log(1-rand()))^(1/shape)
      if (t0+t)<1
        data=vcat(data,1.0*[catnr claims 1 log(t/(1-t)) log(t/(1-t)) 1])
        claims+=1
        scale=scale*cats_scale_update[catnr]
      elseif claims==0
        data[catnr,6]+=1
        covers=vcat(covers,[catnr claims])
      else
        # right censoring here means that there will be no claim before t=1-t0
        data=vcat(data,1.0*[catnr claims 2 log((1-t0)/t0) 1E18 1])
        covers=vcat(covers,[catnr claims])
      end
      t0+=t
    end  
  end
end

# standardise claim count history
mean_claim_hist=mean(data[:,2])
sd_claim_hist=std(data[:,2])
claim_hist_s=(data[:,2].-mean_claim_hist)./sd_claim_hist

# standardise time
hlp_time=data[(ncats+1):size(data,1),4]
mean_time=mean(hlp_time)
sd_time=std(hlp_time)
time_1_s=(data[:,4].-mean_time)./sd_time
time_2_s=(data[:,5].-mean_time)./sd_time

# create hot-encoding of categories
weights=data[:,6]
hots=1.0*(1.0*collect(range(1,ncats)).==permutedims(data[:,1]))'
hotvals=sqrt.(sum(hots.*weights,dims=1)/sum(weights).*((1).-sum(hots.*weights,dims=1)/sum(weights)))    
hots=hots.*hotvals

data_s=[hots claim_hist_s data[:,3] time_1_s time_2_s weights]

# fit model
nvars=ncats+3
nevents=1
layers1=[4,4]
layers2=[4,4]
w,b,npars=initSrvDist(nvars,nevents,layers1,layers2)

iterations=10000
events=[1.0]
w,b,llv=fitSrvDist(w,b,data_s,events,layers1,layers2,iterations,hotvars=[ncats],obsweights=true,c=5.0)

# plot vectors for distributions and hazards
x=0.001*collect(range(1,999))
y1=zeros(ncats,999)
y2=zeros(ncats,999)
y3=zeros(ncats,999)
z1=zeros(ncats,999)
z2=zeros(ncats,999)
z3=zeros(ncats,999)
for catnr=1:ncats
  hotcats=(collect(range(1,ncats)).==catnr)'.*hotvals
  limp=evalSrvPDF([hotcats (0-mean_claim_hist)/sd_claim_hist 1.0 0.0],events,layers1,layers2,w,b)[1]
  inf,sup=limitsSrvCDF(1.0,events,layers1,layers2,w,b)
  x=0.001*collect(range(1,999))  
  for i=1:length(x)
    t=(log(x[i]/(1-x[i]))-mean_time)/sd_time
    y1[catnr,i]=limp*evalSrvPDF(1.0*[hotcats (0-mean_claim_hist)/sd_claim_hist 1 t],events,layers1,layers2,w,b)[2]/(sd_time*x[i]*(1-x[i])*(sup-inf))
    y2[catnr,i]=limp*(evalSrvCDF(1.0*[hotcats (0-mean_claim_hist)/sd_claim_hist 1 t],events,layers1,layers2,w,b)[2]-inf)/(sup-inf)
    y3[catnr,i]=y1[catnr,i]/(1-y2[catnr,i])
    z1[catnr,i]=cats_shape[catnr]/cats_scale[catnr]*(x[i]/cats_scale[catnr])^(cats_shape[catnr]-1)*exp(-(x[i]/cats_scale[catnr])^cats_shape[catnr])
    z2[catnr,i]=1-exp(-(x[i]/cats_scale[catnr])^cats_shape[catnr])
    z3[catnr,i]=z1[catnr,i]/(1-z2[catnr,i])
  end
end

# plot probability density function
p1=plot(x,y1[1,:],st=:line,color="purple",width=2,ylims=(0,2),label="est. Exponential (λ=1)")
p2=plot!(p1,x,z1[1,:],st=:line,line=:dot,color="purple",width=2,label="true Exponential (λ=1)")
p3=plot!(p2,x,y1[2,:],st=:line,color="green",width=2,label="est. Weibull (λ=1,k=0.5)")
p4=plot!(p3,x,z1[2,:],st=:line,line=:dot,color="green",width=2,label="true Weibull (λ=1,k=0.5)")
p5=plot!(p4,x,y1[3,:],st=:line,color="red",width=2,label="est. Weibull (λ=1,k=1.5)")
p6=plot!(p5,x,z1[3,:],st=:line,line=:dot,color="red",width=2,label="true Weibull (λ=1,k=1.5)")
plot(p6)
savefig(p6,"pdf_results.pdf")

# plot cumulative distribution function
p1=plot(x,y2[1,:],st=:line,color="purple",width=2,ylims=(0,0.75),label="est. Exponential (λ=1)")
p2=plot!(p1,x,z2[1,:],st=:line,line=:dot,color="purple",width=2,label="true Exponential (λ=1)")
p3=plot!(p2,x,y2[2,:],st=:line,color="green",width=2,label="est. Weibull (λ=1,k=0.5)")
p4=plot!(p3,x,z2[2,:],st=:line,line=:dot,color="green",width=2,label="true Weibull (λ=1,k=0.5)")
p5=plot!(p4,x,y2[3,:],st=:line,color="red",width=2,label="est. Weibull (λ=1,k=1.5)")
p6=plot!(p5,x,z2[3,:],st=:line,line=:dot,color="red",width=2,label="true Weibull (λ=1,k=1.5)")
plot(p6)
savefig(p6,"cdf_results.pdf")

# plot hazard function
p1=plot(x,y3[1,:],st=:line,color="purple",width=2,ylims=(0,2),label="est. Exponential (λ=1)")
p2=plot!(p1,x,z3[1,:],st=:line,line=:dot,color="purple",width=2,label="true Exponential (λ=1)")
p3=plot!(p2,x,y3[2,:],st=:line,color="green",width=2,label="est. Weibull (λ=1,k=0.5)")
p4=plot!(p3,x,z3[2,:],st=:line,line=:dot,color="green",width=2,label="true Weibull (λ=1,k=0.5)")
p5=plot!(p4,x,y3[3,:],st=:line,color="red",width=2,label="est. Weibull (λ=1,k=1.5)")
p6=plot!(p5,x,z3[3,:],st=:line,line=:dot,color="red",width=2,label="true Weibull (λ=1,k=1.5)",legend=:top)
plot(p6)
savefig(p6,"hazard_results.pdf")

# claim counts observations
counts=zeros(3,20)
for i=1:size(covers,1)
  counts[round(Int,covers[i,1]),1+round(Int,covers[i,2])]+=1
end

# neural simulation claim counts, mean of 100 samples
counts_sim=zeros(3,20)
for catnr=1:ncats
  hotcats=(collect(range(1,ncats)).==catnr)'.*hotvals
  for i=1:100*cats_n[catnr]
    claims=0
    t0=0
    while t0<1
      limp=evalSrvPDF([hotcats (claims-mean_claim_hist)/sd_claim_hist 1.0 0.0],events,layers1,layers2,w,b)[1]
      if limp>rand()
        t=1/(1+exp(-(mean_time+sd_time*quantileSrvCDF([hotcats (claims-mean_claim_hist)/sd_claim_hist 1.0],rand(),events,layers1,layers2,w,b))))
      else
        t=2
      end  
      claims+=1
      t0+=t
    end  
    counts_sim[catnr,claims]+=0.01
  end
end

# plot claim counts

# category 1
p1=plot(collect(range(0,19)),counts[1,:],xlims=(-1,9),xticks=0:1:9,ylims=(0,12000),st=:bar,color="orange",label="observations",ylabel="count")
p2=plot!(p1,collect(range(0,19)),counts_sim[1,:],st=:scatter,color="blue",label="model simulation",xlabel="claims",
yformatter=:plain,title="Exponential interarrival (λ=1)")
plot(p2)
savefig(p2,"claim_counts_1.pdf")

# category 2
p1=plot(collect(range(0,19)),counts[2,:],xlims=(-1,9),xticks=0:1:9,ylims=(0,12000),st=:bar,color="orange",label="observations",ylabel="count")
p2=plot!(p1,collect(range(0,19)),counts_sim[2,:],st=:scatter,color="blue",label="model simulation",xlabel="claims",
yformatter=:plain,title="Weibull interarrival (λ=1,k=0.5)")
plot(p2)
savefig(p2,"claim_counts_2.pdf")

# category 3
p1=plot(collect(range(0,19)),counts[3,:],xlims=(-1,9),xticks=0:1:9,ylims=(0,12000),st=:bar,color="orange",label="observations",ylabel="count")
p2=plot!(p1,collect(range(0,19)),counts_sim[3,:],st=:scatter,color="blue",label="model simulation",xlabel="claims",
yformatter=:plain,title="Weibull interarrival (λ=1,k=1.5)")
plot(p2)
savefig(p2,"claim_counts_3.pdf")

