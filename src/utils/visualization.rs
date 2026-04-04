//! Visualization utilities for plotting data, decision boundaries, and training curves

use plotters::prelude::*;
use std::path::Path;

/// Visualization utilities
pub struct Visualization;

impl Visualization {
    /// Plot 2D scatter plot with optional class labels
    pub fn scatter_plot_2d(
        data: &[(f32, f32)],
        labels: Option<&[usize]>,
        title: &str,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let (x_min, x_max) = data
            .iter()
            .map(|(x, _)| *x)
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), x| {
                (min.min(x), max.max(x))
            });
        let (y_min, y_max) = data
            .iter()
            .map(|(_, y)| *y)
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), y| {
                (min.min(y), max.max(y))
            });

        let x_range = x_max - x_min;
        let y_range = y_max - y_min;
        let margin = 0.1;

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(
                (x_min - margin * x_range)..(x_max + margin * x_range),
                (y_min - margin * y_range)..(y_max + margin * y_range),
            )?;

        chart.configure_mesh().draw()?;

        let colors = [&RED, &BLUE, &GREEN, &CYAN, &MAGENTA, &BLACK];

        if let Some(labels) = labels {
            for (&(x, y), &label) in data.iter().zip(labels.iter()) {
                let color = colors[label % colors.len()];
                chart.draw_series(PointSeries::of_element(
                    vec![(x, y)],
                    5,
                    color,
                    &|c, s, st| Circle::new(c, s, st.filled()),
                ))?;
            }
        } else {
            chart.draw_series(PointSeries::of_element(
                data.iter().cloned(),
                5,
                &BLUE,
                &|c, s, st| Circle::new(c, s, st.filled()),
            ))?;
        }

        root.present()?;
        Ok(())
    }

    /// Plot decision boundary for 2D classification
    pub fn plot_decision_boundary<F>(
        data: &[(f32, f32)],
        labels: &[usize],
        classifier: F,
        title: &str,
        output_path: &Path,
        resolution: usize,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: Fn(f32, f32) -> usize,
    {
        let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let (x_min, x_max) = data
            .iter()
            .map(|(x, _)| *x)
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), x| {
                (min.min(x), max.max(x))
            });
        let (y_min, y_max) = data
            .iter()
            .map(|(_, y)| *y)
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), y| {
                (min.min(y), max.max(y))
            });

        let margin = 0.1;
        let x_range = x_max - x_min;
        let y_range = y_max - y_min;
        let plot_x_min = x_min - margin * x_range;
        let plot_x_max = x_max + margin * x_range;
        let plot_y_min = y_min - margin * y_range;
        let plot_y_max = y_max + margin * y_range;

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(plot_x_min..plot_x_max, plot_y_min..plot_y_max)?;

        chart.configure_mesh().draw()?;

        let step_x = (plot_x_max - plot_x_min) / resolution as f32;
        let step_y = (plot_y_max - plot_y_min) / resolution as f32;

        let boundary_colors = [&RGBColor(255, 200, 200), &RGBColor(200, 200, 255)];

        for i in 0..resolution {
            for j in 0..resolution {
                let x = plot_x_min + i as f32 * step_x;
                let y = plot_y_min + j as f32 * step_y;
                let prediction = classifier(x, y);

                let color = boundary_colors[prediction % boundary_colors.len()];
                chart.draw_series(PointSeries::of_element(
                    vec![(x, y)],
                    1,
                    color,
                    &|c, _, st| {
                        Rectangle::new(
                            [
                                (c.0 - step_x / 2.0, c.1 - step_y / 2.0),
                                (c.0 + step_x / 2.0, c.1 + step_y / 2.0),
                            ],
                            st.filled(),
                        )
                    },
                ))?;
            }
        }

        let colors = [&RED, &BLUE, &GREEN, &CYAN, &MAGENTA, &BLACK];
        for (&(x, y), &label) in data.iter().zip(labels.iter()) {
            let color = colors[label % colors.len()];
            chart.draw_series(PointSeries::of_element(
                vec![(x, y)],
                8,
                color,
                &|c, s, st| Circle::new(c, s, st.filled()),
            ))?;
        }

        root.present()?;
        Ok(())
    }

    /// Plot training loss curve
    pub fn plot_loss_curve(
        losses: &[f32],
        title: &str,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_loss = losses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_loss = losses.iter().cloned().fold(f32::INFINITY, f32::min);

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0f32..losses.len() as f32, min_loss..max_loss)?;

        chart.configure_mesh().draw()?;

        chart
            .draw_series(LineSeries::new(
                losses.iter().enumerate().map(|(i, &loss)| (i as f32, loss)),
                &RED,
            ))?
            .label("Loss")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));

        chart.configure_series_labels().draw()?;
        root.present()?;
        Ok(())
    }
}
