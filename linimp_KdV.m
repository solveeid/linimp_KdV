% Implementation of six different schemes applied to the Korteweg-de Vries
% equation, as it is given in Eidnes and Li: "Linearly implicit local and
% global energy-preserving methods for Hamiltonian PDEs" [EL19].
% The schemes are:
% ** the LILEP and LIGEP schemes of [EL19],
% ** the LEP and GEP schemes of
% Gong, Cai and Wang: "Some new structure-preserving algorithms for general
% multi-symplectic formulations of Hamiltonian PDEs" (2014)
% ** the implicit midpoint (IMP) and multi-symplectic box (MSB) schemes of
% Ascher and McLachlan: "On symplectic and multisymplectic schemes for the
% KdV equation" (2005)
% 
% Code written by Sølve Eidnes and Lu Li

k = .01; % Time step size
T = 10; % End time
N = round(T/k); % Number of time steps

% % Initial conditions:
%
% Test problem 1 in [EL19]:
P = 2; % Length of spatial domain
M = 100; % Number of spatial steps
h = P/M; % Spatial step size
x = (0:h:P-h)';
gamma = 0.022;
eta = 1;
u0 = cos(pi*x);
%
% % Test problem 2 in [EL19]:
% P = 20; % Length of spatial domain
% M = 100; % Number of spatial steps
% h = P/M; % Spatial step size
% x = (0:h:P-h)';
% gamma = 1;
% eta = 6;
% c = 4;
% u_ana = @(x,t) 0.5*c*sech(mod(-x+c*t,P)-P/2).^2;
% u0 = u_ana(x,0);

% Creating the average and forward difference matrices:
e = ones(M,1);
Ax = .5*spdiags([e e], 0:1, M, M);
Ax(M,1) = .5;
deltax = 1/h*spdiags([-e e], 0:1, M, M);
deltax(M,1) = 1/h;
I = speye(M);

uNs = zeros(6,M); % To store solution values from all six methods
% To store discrete approximations to the energy at every time step:
energies = zeros(1,N);
energiesK = zeros(1,N);

% LILEP
u = u0;
tic
for i = 1:N
    un = (1/k*Ax^3 + .5*eta*deltax*Ax*spdiags(Ax*u,0,M,M)*Ax + .5*gamma^2*deltax^3)\(1/k*Ax^3*u-.5*gamma^2*deltax^3*u);
    energies(i) = h*1/6*sum(-gamma^2*3*(deltax*u).^2 + eta*(Ax*u).^3);
    energiesK(i) = h*1/6*sum(-gamma^2*((deltax*u).^2 + 2*(deltax*u).*(deltax*un)) + eta*(Ax*u).^2.*(Ax*un));
    u = un;
end
fprintf('The LILEP scheme used %f seconds.\n',toc);
uNs(1,:) = u;
figure(1)
plot(x,uNs(1,:),'LineWidth',1.5,'Color',[0,.4,1])
hold on

% LEP
u = u0;
tic
for i = 1:N
    un = newton_KdV_LEP(u,u,eta,gamma,deltax,Ax,k,M,1e-10);
    energies(i) = h*1/6*sum(-gamma^2*3*(deltax*u).^2 + eta*(Ax*u).^3);
    energiesK(i) = h*1/6*sum(-gamma^2*((deltax*u).^2 + 2*(deltax*u).*(deltax*un)) + eta*(Ax*u).^2.*(Ax*un));
    u = un;
end
fprintf('The LEP scheme used %f seconds.\n',toc);
uNs(2,:) = u;
figure(1)
plot(x,uNs(2,:),'LineStyle','--','LineWidth',1.5,'Color',[.6,.8,1])

% LIGEP
deltacx = .5/h*spdiags([-e e], [-1,1], M, M);
deltacx(M,1) = .5/h;
deltacx(1,M) = -.5/h;
u = u0;
%D1 = deltacx;
D1 = fourierD1(M,P);
tic
for i = 1:N
    un = (I/k + .5*eta*D1*spdiags(u,0,M,M) + .5*gamma^2*D1^3)\(1/k*u - .5*gamma^2*D1^3*u);
    energies(i) = h*1/6*(-3*gamma^2*sum((D1*un).^2) + eta*sum(un.^3));
    energiesK(i) = h*1/6*(-gamma^2*sum((D1*u).^2+2*(D1*u).*(D1*un)) + eta*sum(u.^2.*un));
    u = un;
end
fprintf('The LIGEP scheme used %f seconds.\n',toc);
uNs(3,:) = u;
figure(1)
plot(x,uNs(3,:),'LineWidth',1.5,'Color',[.8,0,.2])

% GEP
u = u0;
%D1 = deltacx;
D1 = fourierD1(M,P);
tic
for i = 1:N
    un = newton_KdV_GEP(u,u,eta,gamma,D1,k,M,1e-10);
    energies(i) = h*1/6*(-3*gamma^2*sum((D1*un).^2) + eta*sum(un.^3));
    energiesK(i) = h*1/6*(-gamma^2*sum((D1*u).^2+2*(D1*u).*(D1*un)) + eta*sum(u.^2.*un));
    u = un;
end
fprintf('The GEP scheme used %f seconds.\n',toc);
uNs(4,:) = u;
figure(1)
plot(x,uNs(4,:),'LineStyle','--','LineWidth',1.5,'Color',[1,.6,.7])

% Creating matrices needed for the IMP and MSB schemes:
deltac = .5/h*spdiags([-e e], [-1 1], M, M);
deltac(M,1) = .5/h; deltac(1,M) = -.5/h;
dc3 = spdiags([-e 2*e -2*e e], [-2 -1 1 2], M, M);
dc3(M,1) = -2; dc3(1,M) = 2; dc3(M,2) = 1;
dc3(M-1,1) = 1; dc3(1,M-1) = -1; dc3(2,M) = -1;
dc3 = .5/h^3*dc3;
otto = spdiags([e 3*e 3*e e], [-2 -1 0 1], M, M);
otto(1,M-1) = 1; otto(1,M) = 3; otto(2,M) = 1; otto(M,1) = 1;
motto = spdiags([-e 3*e -3*e e], [-2 -1 0 1], M, M);
motto(1,M-1) = -1; motto(1,M) = 3; motto(2,M) = -1; motto(M,1) = 1;
moo = spdiags([-e -e e e], [-2 -1 0 1], M, M);
moo(1,M-1) = -1; moo(1,M) = -1; moo(2,M) = -1; moo(M,1) = 1;

% IMP
u = u0;
energies = zeros(1,N);
energiesK = zeros(1,N);
%theta = 2/3;
theta = 1;
tic
for i = 1:N
    u = newton_KdV_IMP(u,u,eta,gamma,deltac,dc3,k,theta,M,I,1e-10);
end
fprintf('The IMP scheme used %f seconds.\n',toc);
uNs(5,:) = u;
figure(1)
plot(x,uNs(5,:),'LineWidth',1.5,'Color',[.6,.8,.6])

% MSB
u = u0;
tic
for i = 1:N
    u = newton_KdV_MSB(u,u,eta,gamma,otto,motto,moo,k,h,M,1e-10);
end
fprintf('The MSB scheme used %f seconds.\n',toc);
uNs(6,:) = u;
figure(1)
plot(x,uNs(6,:),'LineWidth',1.5,'Color',[.7,.6,.8])
hold off

figure(1)
set(gca,'fontsize',9)
legend('LILEP','LEP','LIGEP','GEP','IMP','MSB','Location','northeast')
ylabel('u(x,T)')
xlabel('x')

%%%%
function un = newton_KdV_LEP(u,un,eta,gamma,deltax,Ax,k,M,tol)
f = @(un) 1/k*Ax^3*(un-u) + 1/6*deltax*Ax*eta*((Ax*un).^2+(Ax*un).*(Ax*u)+(Ax*u).^2) + .5*gamma^2*(deltax^3*un+deltax^3*u);
J = @(un) 1/k*Ax^3 + 1/6*deltax*Ax*eta*(spdiags(Ax*u,0,M,M)*Ax+2*spdiags(Ax*un,0,M,M)*Ax) + .5*gamma^2*deltax^3;
err = norm(f(un));
c = 0;
while err > tol
    un = un - J(un)\f(un);
    err = norm(f(un));
    c = c+1;
    if c > 5
        break;
        err
    end
end
end

function un = newton_KdV_GEP(u,un,eta,gamma,Dx,k,M,tol)
f = @(un) 1/k*(un-u) + 1/6*Dx*eta*(un.^2+un.*u+u.^2) + .5*gamma^2*(Dx^3*un+Dx^3*u);
I = speye(M);
J = @(un) 1/k*I + 1/6*Dx*eta*(spdiags(u,0,M,M)+2*spdiags(un,0,M,M)) + .5*gamma^2*Dx^3;
err = norm(f(un));
c = 0;
while err > tol
    un = un - J(un)\f(un);
    err = norm(f(un));
    c = c+1;
    if c > 5
        break;
        err
    end
end
end

function un = newton_KdV_IMP(u,un,eta,gamma,deltac,dc3,k,theta,M,I,tol)
f = @(un) 1/k*(un-u) + .5*eta*(theta*deltac*((.5*(u+un)).^2) + (1-theta)*(u+un).*(deltac*.5*(u+un))) + gamma^2*dc3*(.5*(u+un));
J = @(un) 1/k*I + .5*eta*(.5*theta*deltac*spdiags(u+un,0,M,M) + (1-theta)*.5*(spdiags(u+un,0,M,M)*deltac + spdiags(deltac*(u+un),0,M,M))) + .5*gamma^2*dc3;
err = norm(f(un));
c = 0;
while err > tol
    un = un - J(un)\f(un);
    err = norm(f(un));
    c = c+1;
    if c > 5
        break;
        err
    end
end
end

function un = newton_KdV_MSB(u,un,eta,gamma,otto,motto,moo,k,h,M,tol)
f = @(un) 1/k*otto*(un-u) + .25*eta/h*moo*((u+un).^2) + 4*gamma^2/h^3*motto*(u+un);
J = @(un) 1/k*otto + .5*eta/h*moo*spdiags(u+un,0,M,M) + 4*gamma^2/h^3*motto;
err = norm(f(un));
c = 0;
while err > tol
    un = un - J(un)\f(un);
    err = norm(f(un));
    c = c+1;
    if c > 5
        break;
        err
    end
end
end

function D1 = fourierD1(M,L)
mu = 2*pi/L;
D1 = zeros(M,M);
for j = 1:M
    for k = 1:M
        if j == k
            D1(j,k) = 0;
        else
            D1(j,k) = 1/2*mu*(-1)^(j+k)*cot(mu*(j-k)*L/M/2);
        end
    end
end
if mod(M,2) == 1
    for j = 1:(M-1)/2
        for k = (j+(M+1)/2):M
            D1(j,k) = -D1(j,k);
            D1(M+1-j,M+1-k) = -D1(M+1-j,M+1-k);
        end
    end
end
end