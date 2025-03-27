/*
 * Author:
 * Copyright (c) 2025/02/05- Yuqing Liang (BIMCoder Liang)
 * bim.frankliang@foxmail.com
 *
 * Use of this source code is governed by a GPL-3.0 license that can be found in
 * the LICENSE file.
 *
 */

#include "ND_CurveEvalModule.h"
#include "ND_SurfaceEvalModule.h"
#include "XYZ.h"
#include "XYZW.h"
#include "Constants.h"
#include "LNObject.h"
#include "NurbsSurface.h"

#include <vtkNew.h>
#include <vtkAxesActor.h>
#include <vtkActor.h>
#include <vtkTransform.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkInteractorStyleTrackballActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkAutoInit.h>
#include <vtkCallbackCommand.h>
#include <vtkRendererCollection.h>

#include "customIteractorStyle.h"
#include "customVtkHelper.h"

#include <vector>
#include <cmath>

using namespace LNLib;
VTK_MODULE_INIT(vtkRenderingOpenGL2)

#pragma region Fitting Module
void CurveFitting(vtkSmartPointer<vtkRenderer> renderer)
{
    int num_ctrl_pts = 64;
    int degree = 3;
    int num_eval_pts = 256;
    int epochs = 10000;
    double lr = 0.1;

    std::vector<LNLib::XYZ> targetPoints(num_eval_pts);
    double pStep = 2 * LNLib::Constants::Pi / (double)(num_eval_pts - 1);
    for (int i = 0; i < num_eval_pts; i++)
    {
        double x = 0 + i * pStep;
        double y = std::sin(x) + 2 * std::sin(2 * x) + sin(4 * x);
        double z = 0;
        targetPoints[i] = LNLib::XYZ(x, y, z);
    }

    DisplayTargetCurve(renderer, targetPoints);

    torch::Tensor target_tensor = torch::zeros({ 1, num_eval_pts, 3 });
    for (int i = 0; i < num_eval_pts; ++i) {
        target_tensor[0][i][0] = targetPoints[i].X();
        target_tensor[0][i][1] = targetPoints[i].Y();
        target_tensor[0][i][2] = targetPoints[i].Z();
    }

    // Sample from the target point, to initialize the control point
    std::vector<LNLib::XYZ> ctrl_points(num_ctrl_pts);
    double sample_step = (num_eval_pts - 1) / double(num_ctrl_pts - 1);
    for (int i = 0; i < num_ctrl_pts; ++i) {
        int idx = std::round(i * sample_step);
        ctrl_points[i] = targetPoints[std::min(idx, num_eval_pts - 1)];
    }

    // Create tensor
    torch::Tensor ctrl_pts_tensor = torch::zeros({ 1, num_ctrl_pts, 3 });
    for (int i = 0; i < num_ctrl_pts; ++i) {
        ctrl_pts_tensor[0][i][0] = ctrl_points[i].X();
        ctrl_pts_tensor[0][i][1] = ctrl_points[i].Y();
        ctrl_pts_tensor[0][i][2] = ctrl_points[i].Z();
    }
    ctrl_pts_tensor = ctrl_pts_tensor.requires_grad_(true); // CPU

    ND_LNLib::CurveEvalModule cm(num_ctrl_pts, degree, num_eval_pts);

    std::vector<torch::Tensor> group{ ctrl_pts_tensor };
    torch::optim::AdamOptions options(lr);
    torch::optim::Adam opt(group, options);
    torch::optim::ReduceLROnPlateauScheduler scheduler(opt, torch::optim::ReduceLROnPlateauScheduler::SchedulerMode::min);

    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        opt.zero_grad();
        torch::Tensor weights = torch::ones({ 1, num_ctrl_pts, 1 }); // CPU
        torch::Tensor pred = cm.forward(torch::cat({ ctrl_pts_tensor, weights }, -1));

        auto dist = torch::cdist(pred, target_tensor);
        auto loss1 = torch::mean(torch::amin(dist, 2));
        auto loss2 = torch::mean(torch::amin(dist, 1));
        auto chamfer_loss = (loss1 + loss2) / 2.0;

        torch::Tensor total_loss = chamfer_loss;
        if (epoch < 3000) {
            auto diff = pred.slice(1, 0, -1) - pred.slice(1, 1);
            auto curve_length = torch::mean(torch::sum(diff.square(), { 1,2 }));
            total_loss += curve_length * 0.1;
        }

        total_loss.backward();
        opt.step();

        scheduler.step(total_loss.item<float>());

        if (epoch % 1000 == 0) {

            double curveLoss = total_loss.item<double>();

            // Save control points
            std::vector<XYZW> fittingControlPoints;
            auto ctrl_data = ctrl_pts_tensor.accessor<float, 3>();
            for (int i = 0; i < ctrl_data.size(1); ++i) {
                double x = ctrl_data[0][i][0];
                double y = ctrl_data[0][i][1];
                double z = ctrl_data[0][i][2];

                XYZW fp = XYZW(XYZ(x, y, z), 1);
                fittingControlPoints.emplace_back(fp);
            }

            int m = num_ctrl_pts + degree;
            std::vector<double> fittingKnotVector(m + 1);

            for (int i = 0; i <= m; ++i) {
                if (i <= degree) {
                    fittingKnotVector[i] = 0.0;
                }
                else if (i >= m - degree) {
                    fittingKnotVector[i] = 1.0;
                }
                else {
                    fittingKnotVector[i] = static_cast<double>(i - degree) / (num_ctrl_pts - degree);
                }
            }

            LN_NurbsCurve curve;
            curve.Degree = degree;
            curve.KnotVector = fittingKnotVector;
            curve.ControlPoints = fittingControlPoints;

            DisplayNurbsCurve(renderer, curve);

            // Save prediction points
            auto pred_data = pred.accessor<float, 3>();
            for (int i = 0; i < pred_data.size(1); ++i) {
                double x = pred_data[0][i][0];
                double y = pred_data[0][i][1];
                double z = pred_data[0][i][2];
            }
        }
    }    
}

void SurfaceFitting(vtkSmartPointer<vtkRenderer> renderer)
{
    int degreeU = 2;
    int degreeV = 2;
    std::vector<double> kvU = { 0,0,0,1,2,3,4,4,5,5,5 };
    std::vector<double> kvV = { 0,0,0,1,2,3,3,3 };
    UV uv = UV(5.0 / 2, 1);

    XYZW P20 = XYZW(-1, 2, 4, 1);
    XYZW P21 = XYZW(0, 2, 4, 1);
    XYZW P22 = XYZW(0, 6, 4, 2);
    XYZW P23 = XYZW(0, 2, 0, 1);
    XYZW P24 = XYZW(1, 2, 0, 1);

    XYZW P10 = 0.9 * P20;
    XYZW P11 = 0.9 * P21;
    XYZW P12 = 0.9 * P22;
    XYZW P13 = 0.9 * P23;
    XYZW P14 = 0.9 * P24;

    XYZW P00 = 0.9 * P10;
    XYZW P01 = 0.9 * P11;
    XYZW P02 = 0.9 * P12;
    XYZW P03 = 0.9 * P13;
    XYZW P04 = 0.9 * P14;

    XYZW P30 = XYZW(3, 6, 8, 2);
    XYZW P31 = XYZW(4, 6, 8, 2);
    XYZW P32 = XYZW(12, 24, 12, 6);
    XYZW P33 = XYZW(4, 6, 0, 2);
    XYZW P34 = XYZW(5, 6, 0, 2);

    XYZW P40 = XYZW(3, 2, 4, 1);
    XYZW P41 = XYZW(4, 2, 4, 1);
    XYZW P42 = XYZW(8, 6, 4, 2);
    XYZW P43 = XYZW(4, 2, 0, 1);
    XYZW P44 = XYZW(5, 2, 0, 1);

    XYZW P50 = 1.5 * P40;
    XYZW P51 = 1.5 * P41;
    XYZW P52 = 1.5 * P42;
    XYZW P53 = 1.5 * P43;
    XYZW P54 = 1.5 * P44;

    XYZW P60 = 1.5 * P50;
    XYZW P61 = 1.5 * P51;
    XYZW P62 = 1.5 * P52;
    XYZW P63 = 1.5 * P53;
    XYZW P64 = 1.5 * P54;

    XYZW P70 = 1.5 * P60;
    XYZW P71 = 1.5 * P61;
    XYZW P72 = 1.5 * P62;
    XYZW P73 = 1.5 * P63;
    XYZW P74 = 1.5 * P64;

    std::vector<std::vector<XYZW>> cps = {

        {P00, P01, P02, P03, P04},
        {P10, P11, P12, P13, P14},
        {P20, P21, P22, P23, P24},
        {P30, P31, P32, P33, P34},
        {P40, P41, P42, P43, P44},
        {P50, P51, P52, P53, P54},
        {P60, P61, P62, P63, P64},
        {P70, P71, P72, P73, P74},

    };

    //Target Surface
    LN_NurbsSurface surface;
    surface.DegreeU = degreeU;
    surface.DegreeV = degreeV;
    surface.KnotVectorU = kvU;
    surface.KnotVectorV = kvV;
    surface.ControlPoints = cps;
    DisplaySurface(renderer, surface, true);

    //Start Fitting
    int num_ctrl_pts_u = 12;
    int num_ctrl_pts_v = 12;
    int num_eval_pts_u = 64;
    int num_eval_pts_v = 64;
    double step_u = (kvU.back() - kvU.front()) / (num_eval_pts_u - 1);
    double step_v = (kvV.back() - kvV.front()) / (num_eval_pts_v - 1);

    std::vector<XYZ> targetPoints;
    for (int i = 0; i < num_eval_pts_u; ++i) {
        double u = kvU.front() + i * step_u;
        for (int j = 0; j < num_eval_pts_v; ++j) {
            double v = kvV.front() + j * step_v;
            targetPoints.push_back(NurbsSurface::GetPointOnSurface(surface, UV(u, v)));
        }
    }

    // transform to tensor [1, num_points, 3]
    torch::Tensor target_tensor = torch::zeros({1, num_eval_pts_u * num_eval_pts_v, 3});
    for (size_t i = 0; i < targetPoints.size(); ++i) {
        target_tensor[0][i][0] = targetPoints[i].GetX();
        target_tensor[0][i][1] = targetPoints[i].GetY();
        target_tensor[0][i][2] = targetPoints[i].GetZ();
    }
    
    // Initialize trainable control points
    auto inp_ctrl_pts = torch::randn({1, num_ctrl_pts_u, num_ctrl_pts_v, 3}, torch::dtype(torch::kDouble).requires_grad(true));

    ND_LNLib::SurfaceEvalModule sm(
        num_ctrl_pts_u, num_ctrl_pts_v,
        3, 3,          // degree U/V
        num_eval_pts_u, num_eval_pts_v,
        3              // dimension
    );

    torch::optim::Adam optimizer({inp_ctrl_pts}, torch::optim::AdamOptions(0.01));

    std::cout << "start training" << std::endl;
    for (int epoch = 0; epoch < 10000; ++epoch) {
        optimizer.zero_grad();
        
        auto weights = torch::ones({1, num_ctrl_pts_u, num_ctrl_pts_v, 1});
        auto input = torch::cat({inp_ctrl_pts, weights}, -1);
        
        auto output = sm.forward(input);
        output = output.reshape({ 1, num_eval_pts_u * num_eval_pts_v, 3 });

        // ¼ÆËãËðÊ§
        torch::Tensor loss = ((target_tensor - output).pow(2)).mean();
        
        loss.backward();
        optimizer.step();

        if ((epoch + 1) % 10 == 0) {
            std::cout << "Epoch " << epoch + 1 
                    << ", Loss: " << loss.item<float>() << std::endl;
        }

    }

    LN_NurbsSurface fittedSurface;
    
    fittedSurface.DegreeU = degreeU;
    fittedSurface.DegreeV = degreeV;
    fittedSurface.KnotVectorU = kvU;
    fittedSurface.KnotVectorV = kvV;
    
    auto ctrl_pts_tensor = inp_ctrl_pts.squeeze(0);
    fittedSurface.ControlPoints.resize(num_ctrl_pts_u);
    for(int i=0; i< num_ctrl_pts_u; ++i){
        fittedSurface.ControlPoints[i].resize(num_ctrl_pts_v);
        for(int j=0; j< num_ctrl_pts_v; ++j){
            auto pt = ctrl_pts_tensor[i][j];
            fittedSurface.ControlPoints[i][j] = XYZW(
                pt[0].item<double>(),  // x
                pt[1].item<double>(),  // y
                pt[2].item<double>(),  // z
                1.0                    // w
            );
        }
    }

    DisplaySurface(renderer, fittedSurface, false);

}
#pragma endregion

int main(int, char* [])
{
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetWindowName("ND_LNLib - BIMCoder Liang (bim.frankliang@foxmail.com)");
    renderWindow->FullScreenOn();
    renderWindow->BordersOn();

    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    
    //Example 1: Curve Fitting:
    //CurveFitting(renderer);
    //or Example 2: Surface Fitting:
    SurfaceFitting(renderer);

    vtkNew<vtkAxesActor> axesActor;
    vtkNew<vtkTransform>  userTrans;
    userTrans->Update();
    axesActor->SetUserTransform(userTrans);
    axesActor->AxisLabelsOn();
    axesActor->SetTotalLength(100, 100, 100);
    renderer->AddActor(axesActor);
    renderer->SetBackground(0, 0, 0);

    vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
    renderWindowInteractor->SetRenderWindow(renderWindow);

    vtkNew<customIteractorStyle> style;
    style->SetFixedAxesActor(axesActor);
    renderWindowInteractor->SetInteractorStyle(style);

    renderWindow->AddRenderer(renderer);
    renderer->ResetCamera();
    renderWindow->Render();

    renderWindowInteractor->Start();
    return 0;
}
