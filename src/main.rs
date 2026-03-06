mod agents;
mod config;
mod env;
mod nn;
mod trainer;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::Colorize;
use std::{fs, path::PathBuf};

use config::Config;

// ─────────────────────────────────────────────────────────
// CLI definition
// ─────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name        = "rl-trainer",
    version     = "0.1.0",
    about       = "Train RL agents from a TOML config file — no code required",
    long_about  = None,
    after_help  = "\
EXAMPLES:\n\
    rl-trainer train -c examples/cartpole_dqn.toml\n\
    rl-trainer train -c config.toml --output ./my_model -v\n\
    rl-trainer eval  -c examples/cartpole_dqn.toml -m ./models/cartpole_dqn\n\
    rl-trainer init  --algorithm ppo --env cartpole -o my_config.toml",
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train an RL agent from a TOML config file
    Train {
        /// Path to the TOML config file
        #[arg(short, long, value_name = "FILE")]
        config: PathBuf,

        /// Override the model output path from config
        #[arg(short, long, value_name = "PATH")]
        output: Option<PathBuf>,

        /// Print per-episode details (verbose)
        #[arg(short, long)]
        verbose: bool,
    },

    /// Evaluate a saved model
    Eval {
        /// Path to the TOML config file (used for environment settings)
        #[arg(short, long, value_name = "FILE")]
        config: PathBuf,

        /// Path to the saved model (with or without .json extension)
        #[arg(short, long, value_name = "PATH")]
        model: PathBuf,

        /// Number of evaluation episodes
        #[arg(short, long, default_value = "10")]
        episodes: usize,
    },

    /// Generate a sample TOML config file
    #[command(name = "init")]
    InitConfig {
        /// Algorithm to generate config for (dqn | reinforce | ppo)
        #[arg(short, long, default_value = "dqn")]
        algorithm: String,

        /// Environment to generate config for (cartpole | mountain_car | gridworld)
        #[arg(short, long, default_value = "cartpole")]
        env: String,

        /// Output path for the generated config
        #[arg(short, long, default_value = "config.toml")]
        output: PathBuf,
    },
}

// ─────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    print_banner();

    match cli.command {
        Commands::Train {
            config,
            output,
            verbose,
        } => {
            let toml_str = fs::read_to_string(&config)
                .with_context(|| format!("Cannot read config file '{}'", config.display()))?;

            let mut cfg: Config = toml::from_str(&toml_str)
                .with_context(|| format!("Failed to parse TOML in '{}'", config.display()))?;

            // CLI override for output path
            if let Some(out) = output {
                cfg.output.model_path = out.to_string_lossy().to_string();
            }

            cfg.validate().with_context(|| "Config validation failed")?;

            print_config_summary(&cfg);

            trainer::train(&cfg, verbose).with_context(|| "Training failed")?;

            println!("\n{}", "✅  Done! Model saved successfully.".green().bold());
        }

        Commands::Eval {
            config,
            model,
            episodes,
        } => {
            let toml_str = std::fs::read_to_string(&config)
                .with_context(|| format!("Cannot read config file '{}'", config.display()))?;

            let cfg: Config = toml::from_str(&toml_str)
                .with_context(|| format!("Failed to parse TOML in '{}'", config.display()))?;

            trainer::evaluate(&cfg, &model.to_string_lossy(), episodes)
                .with_context(|| "Evaluation failed")?;
        }

        Commands::InitConfig {
            algorithm,
            env,
            output,
        } => {
            let sample = config::generate_sample_config(&algorithm, &env);
            std::fs::write(&output, &sample)
                .with_context(|| format!("Cannot write to '{}'", output.display()))?;

            println!(
                "{}",
                format!(
                    "✅  Sample config for {} + {} written to: {}",
                    algorithm.yellow(),
                    env.yellow(),
                    output.display().to_string().bright_white()
                )
                .green()
            );
            println!(
                "\n{}\n{}",
                "── Preview ─────────────────────────────────".cyan(),
                sample
            );
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────
// Aesthetics
// ─────────────────────────────────────────────────────────

fn print_banner() {
    let logo = r#"
████████╗ ██████╗ ██████╗ ██╗
╚══██╔══╝██╔═══██╗██╔══██╗██║
   ██║   ██║   ██║██████╔╝██║
   ██║   ██║   ██║██╔══██╗██║
   ██║   ╚██████╔╝██║  ██║███████╗
   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝
"#;
    println!("{}", logo.bright_cyan());
    println!(
        "  {}",
        "Train RL models from TOML config — zero code required!"
            .bright_white()
            .bold()
    );
    println!(
        "  {}",
        "Algorithms: DQN · REINFORCE · PPO   |   Envs: CartPole · MountainCar · GridWorld"
            .bright_black()
    );
    println!();
}

fn print_config_summary(cfg: &Config) {
    let line = "─".repeat(48);
    println!("{}", format!("── Config {line}").cyan().bold());

    let kv = |k: &str, v: &str| println!("  {:<22} {}", k.bright_black(), v.bright_white());

    kv("Environment:", &cfg.environment.name);
    kv("Max steps:", &cfg.environment.max_steps.to_string());
    kv("Algorithm:", &cfg.algorithm.name.to_uppercase());
    kv(
        "Learning rate:",
        &format!("{:.0e}", cfg.algorithm.learning_rate),
    );
    kv("Gamma:", &format!("{}", cfg.algorithm.gamma));
    kv(
        "Hidden layers:",
        &format!("{:?}", cfg.network.hidden_layers),
    );
    kv("Activation:", &cfg.network.activation);
    kv("Episodes:", &cfg.training.episodes.to_string());
    kv("Output path:", &format!("{}.json", cfg.output.model_path));
    kv("Seed:", &cfg.environment.seed.to_string());

    // Algorithm-specific
    match cfg.algorithm.name.as_str() {
        "dqn" => {
            let d = &cfg.algorithm.dqn;
            kv("Replay buffer:", &d.buffer_size.to_string());
            kv("Batch size:", &d.batch_size.to_string());
            kv("Epsilon start:", &format!("{}", d.epsilon_start));
            kv(
                "Target sync:",
                &format!("every {} steps", d.target_update_freq),
            );
        }
        "ppo" => {
            let p = &cfg.algorithm.ppo;
            kv("Clip ε:", &format!("{}", p.clip_epsilon));
            kv("Update epochs:", &p.epochs.to_string());
            kv("Steps/update:", &p.steps_per_update.to_string());
            kv("GAE λ:", &format!("{}", p.gae_lambda));
        }
        "reinforce" => {
            let r = &cfg.algorithm.reinforce;
            kv("Baseline:", &r.baseline);
            kv("Normalize returns:", &r.normalize_returns.to_string());
        }
        _ => {}
    }

    println!("{}", format!("── Training {line}").cyan().bold());
    println!();
}
