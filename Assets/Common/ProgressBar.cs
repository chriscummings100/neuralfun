using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ProgressBar : MonoBehaviour
{
    public float m_progress = 0.0f;
    public Transform m_bar;
    public TMPro.TextMeshProUGUI m_label;

    int m_last_shown = -1;
    
    // Start is called before the first frame update
    void OnEnable()
    {
        m_last_shown = -1;
        Refresh();
    }

    // Update is called once per frame
    void Update()
    {
        Refresh();
    }

    void Refresh()
    {
        int progress = Mathf.RoundToInt(m_progress * 100);
        if (progress != m_last_shown)
        {
            m_last_shown = progress;
            m_label.text = $"{progress}%";
            m_bar.localScale = new Vector3(m_progress, 1.0f, 1.0f);
        }
    }
}
