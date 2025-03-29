#![allow(unused)]

use std::{io::Write, time::Duration};

use gilrs::{
    Axis,
    Button,
    Gilrs,
    Event,
};

macro_rules! button_id_fns {
    ($(Button::$name:ident => $id:literal),*$(,)?) => {
        pub fn button_id(button: Button) -> u32 {
            match button {
                $(
                    Button::$name => $id,
                )*
                Button::Unknown => 19,
            }
        }

        pub fn button_from_id(id: u32) -> Button {
            match id {
                $(
                    $id => Button::$name,
                )*
                _ => Button::Unknown,
            }
        }
    };
}

button_id_fns!(
    Button::South => 0,
    Button::East => 1,
    Button::North => 2,
    Button::West => 3,
    Button::C => 4,
    Button::Z => 5,
    Button::LeftTrigger => 6,
    Button::LeftTrigger2 => 7,
    Button::RightTrigger => 8,
    Button::RightTrigger2 => 9,
    Button::Select => 10,
    Button::Start => 11,
    Button::Mode => 12,
    Button::LeftThumb => 13,
    Button::RightThumb => 14,
    Button::DPadUp => 15,
    Button::DPadDown => 16,
    Button::DPadLeft => 17,
    Button::DPadRight => 18,
);

pub struct GamepadState {
    button_press_mask: u32,

}

#[test]
fn foo() -> Result<(), gilrs::Error> {
    let mut gamepad = Gilrs::new()?;
    'event_loop: loop {
        while let Some(event) = gamepad.next_event() {
            match event.event {
                gilrs::EventType::ButtonPressed(button, code) => {
                    match button {
                        Button::South => println!("(A) Pressed."),
                        Button::Start => {
                            println!("Exiting...");
                            break 'event_loop;
                        }
                        _ => (),
                    }
                },
                gilrs::EventType::ButtonRepeated(button, code) => (),
                gilrs::EventType::ButtonReleased(button, code) => (),
                gilrs::EventType::ButtonChanged(button, _, code) => (),
                gilrs::EventType::AxisChanged(axis, value, code) => {
                    if axis == Axis::RightZ {
                        println!("RightZ: {value:.3}");
                    }
                },
                gilrs::EventType::Connected => (),
                gilrs::EventType::Disconnected => (),
                gilrs::EventType::Dropped => (),
                gilrs::EventType::ForceFeedbackEffectCompleted => (),
                _ => (),
            }
        }
        spin_sleep::sleep(Duration::from_millis(100));
    }
    Ok(())
}